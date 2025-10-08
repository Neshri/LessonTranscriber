#!/bin/bash

DEST="/mnt/lesson_audio"
LOCKDIR="/tmp/rec.lock"
PIDFILE="$LOCKDIR/pid"
ERROR_LOG="$LOCKDIR/error.log"
REC_FILE="$LOCKDIR/rec.file"
FINAL_FILE="$LOCKDIR/rec.final"

# Function to clean up lock on exit
cleanup() {
    if [ -d "$LOCKDIR" ]; then
        PID=$(cat "$PIDFILE" 2>/dev/null)
        if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
            # Process is still running, don't clean up lock
            return
        fi
        # Process not running, clean up lock
        rm -rf "$LOCKDIR"
    fi
}

# Set trap for cleanup on signals
trap cleanup EXIT INT TERM

# Ensure destination directory exists
if [ ! -d "$DEST" ]; then
    echo "Fel: Destinationsmappen $DEST finns inte."
    exit 1
fi

case "$1" in
  start)
    # Check if already running
    if [ -d "$LOCKDIR" ]; then
        PID=$(cat "$PIDFILE" 2>/dev/null)
        if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
            echo "Fel: En inspelning pågår redan (PID: $PID)."
            exit 1
        else
            # Clean up stale lock
            rm -rf "$LOCKDIR"
        fi
    fi

    # Create lock
    if ! mkdir "$LOCKDIR" 2>/dev/null; then
        echo "Fel: Kunde inte skapa lås."
        exit 1
    fi

    echo "Starting new recording..."
    BASENAME="rec_$(date +%Y-%m-%d_%H-%M-%S)"
    TMPFILE="$DEST/${BASENAME}_recording.mp3"
    FINALFILE="$DEST/${BASENAME}_finished.mp3"

    # Start recording in background with error logging
    bash -c "exec arecord -f cd -t raw -D plughw:2,0 | lame -r - '$TMPFILE'" &> "$ERROR_LOG" &
    PID=$!
    echo $PID > "$PIDFILE"

    # Brief wait and health check
    sleep 2
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "Fel: Inspelningen misslyckades att starta."
        echo "Felmeddelande:"
        cat "$ERROR_LOG" 2>/dev/null
        cleanup
        exit 1
    fi

    # Save files
    echo "$TMPFILE" > "$REC_FILE"
    echo "$FINALFILE" > "$FINAL_FILE"

    echo "Inspelning startad: $TMPFILE"
    echo "Process ID: $PID"
    ;;

  stop)
    if [ ! -d "$LOCKDIR" ]; then
        echo "Fel: Ingen inspelning verkar pågå."
        exit 1
    fi

    PID=$(cat "$PIDFILE" 2>/dev/null)
    if [ -z "$PID" ] || ! kill -0 "$PID" 2>/dev/null; then
        echo "Fel: Ingen giltig process att stoppa."
        cleanup
        exit 1
    fi

    # Gracefully stop
    pkill -TERM -P "$PID" 2>/dev/null
    kill -TERM "$PID" 2>/dev/null

    # Wait with timeout
    timeout=0
    while kill -0 "$PID" 2>/dev/null && [ $timeout -lt 20 ]; do
        sleep 0.5
        timeout=$((timeout + 1))
    done

    if kill -0 "$PID" 2>/dev/null; then
        # Force kill if still running
        pkill -KILL -P "$PID" 2>/dev/null
        kill -KILL "$PID" 2>/dev/null
        sleep 1
    fi

    TMPFILE=$(cat "$REC_FILE" 2>/dev/null)
    FINALFILE=$(cat "$FINAL_FILE" 2>/dev/null)

    if [ -f "$TMPFILE" ]; then
        mv "$TMPFILE" "$FINALFILE"
        echo "Inspelning stoppad och slutförd: $FINALFILE"
    else
        echo "Fel: Ingen inspelningsfil hittades."
    fi

    cleanup
    ;;

  *)
    echo "Användning: $0 {start|stop}"
    exit 1
    ;;
esac