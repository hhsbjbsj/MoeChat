#!/bin/bash

PYTHON_PATH="/home/keruis/extra/MoeChat-main/pp/bin/python"

SCRIPT_1="script1.py"
TITLE_1="PythonScript1"

SCRIPT_2="script2.py"
TITLE_2="PythonScript2"

SCRIPT_3="script3.py"
TITLE_3="PythonScript3"

cd "$(dirname "$0")"

toggle_script() {
    local title="$1"
    local script="$2"

    pid=$(pgrep -f "$PYTHON_PATH $script")
    if [ -n "$pid" ]; then
        echo "Found $title running (PID: $pid). Killing..."
        kill "$pid"
    else
        echo "Starting $title..."
        nohup "$PYTHON_PATH" "$script" > "$title.log" 2>&1 &
    fi
}

toggle_script "$TITLE_1" "$SCRIPT_1"
toggle_script "$TITLE_2" "$SCRIPT_2"
toggle_script "$TITLE_3" "$SCRIPT_3"

echo "Done."