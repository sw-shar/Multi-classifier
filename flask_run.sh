#!/bin/bash

set -eu -o pipefail

[ $# = 0 ]

is_debug=${RUN_DEBUG:-}
# "" -- false
# "1" -- true

export FLASK_APP=app

if [ "$is_debug" ]; then
	export FLASK_ENV=development
fi

exec flask run --with-threads --host=0.0.0.0
