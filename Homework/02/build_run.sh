#!/bin/bash

# =============================================
# INITIALIZATION
# =============================================

# Define log file path (from .env or default)
LOG_FILE="${TRAIN_LOG_FILE:-run_container.log}"

# Overwrite existing log file and start fresh
echo "=== Starting Docker Container Run ===" > "$LOG_FILE"
echo "Timestamp: $(date)" >> "$LOG_FILE"

# Function to log messages to both console and log file
log() {
    echo "$1"
    echo "$1" >> "$LOG_FILE"
}

# =============================================
# ENVIRONMENT SETUP
# =============================================

log "Loading environment variables..."

# Load environment variables from .env file
if [ -f .env ]; then
    log "Found .env file, loading variables..."
    # Use grep to filter out comments and empty lines, then export
    ENV_VARS=$(grep -v '^#' .env | grep -v '^$')
    echo "$ENV_VARS" >> "$LOG_FILE"
    export $(echo "$ENV_VARS" | xargs)
else
    log "Error: .env file not found in current directory"
    exit 1
fi

# Log all environment variables that start with MLFLOW_ or BENTOML_
log "Relevant environment variables:"
env | grep -E '^(MLFLOW_|BENTOML_|TRAIN_|CONTAINER_)' >> "$LOG_FILE"

# =============================================
# CONTAINER OPERATIONS
# =============================================

# Define variables from .env
IMAGE_NAME="${TRAIN_IMAGE_NAME:-data-preprocessing}"
DOCKERFILE="Dockerfile"
OUTPUT_DIR="${CONTAINER_APP_OUTPUT_DIR:-./outputs}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CONTAINER_NAME="${TRAIN_CONTAINER_NAME_PREFIX:-container}-${TIMESTAMP}"

# Create outputs directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Build the Docker image
log "Building Docker image ${IMAGE_NAME}..."
docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" . >> "$LOG_FILE" 2>&1

# Run the container with full logging
log "Running container ${CONTAINER_NAME}..."
log "=================== CONTAINER OUTPUT ==================="
docker run --name "$CONTAINER_NAME" \
  -v "$(pwd)/outputs:${CONTAINER_APP_OUTPUT_DIR}" \
  -e MLFLOW_TRACKING_URI \
  -e MLFLOW_TRACKING_USERNAME \
  -e MLFLOW_TRACKING_TOKEN \
  -e MLFLOW_TRACKING_PASSWORD \
  "$IMAGE_NAME" \
  --raw_data_path "$RAW_DATA_PATH" \
  --dest_path "$DEST_PATH" >> "$LOG_FILE" 2>&1

# Check container exit status
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    log "Container exited with error code ${EXIT_CODE}. Check ${LOG_FILE} for details."
else
    log "Container completed successfully."
fi

# =============================================
# AGGRESSIVE CLEANUP
# =============================================

log "Performing aggressive Docker cleanup..."

# Stop and remove the specific container we just ran
docker rm -f "$CONTAINER_NAME" 2>/dev/null || log "Container ${CONTAINER_NAME} already removed"

# Remove all stopped containers
docker container prune -f >> "$LOG_FILE" 2>&1

# Remove all unused images (including dangling ones)
docker image prune -a -f >> "$LOG_FILE" 2>&1

# Remove all unused networks
docker network prune -f >> "$LOG_FILE" 2>&1

# Remove all unused volumes
docker volume prune -f >> "$LOG_FILE" 2>&1

log "Cleanup complete."
log "Full logs available in: ${LOG_FILE}"