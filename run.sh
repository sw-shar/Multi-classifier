#!/bin/bash

set -eux -o pipefail

[ $# = 2 ]
image=$1
container=$2

is_debug=${RUN_DEBUG:-}
# "" -- false
# "1" -- true

opts=(
        -it
        --rm
	-p 5000:5000
        --name="$container"
)
cmd=()

if [ "$is_debug" ]; then
	opts+=(
		-v "$PWD":/app
	)
	cmd=(
		env RUN_DEBUG=1 ./flask_run.sh
	)
fi
        
set -x
docker run "${opts[@]}" "$image" "${cmd[@]}"
