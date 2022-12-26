#!/bin/bash

set -e -o pipefail

[ $# = 2 ]
image=$1
code=$2

log=tmp/curl-json.log

case $code in
	200) ;;
	400) export KEY_IMAGE=kartinka ;;
	503) image=/dev/null ;;
	*) exit 1
esac

args_curl=(
	-X POST
       	-H 'Content-Type: application/json' 
	--data @-
	http://localhost:5000/forward
)

set +e  # don't exit on error
./image2json.sh "$image" |
	curl "${args_curl[@]}" |
	tee "$log" |
       	jq -r '.image' |
	base64 -d - | 
	display -

if [ "$code" != 200 ]; then
	cat "$log"
	echo
fi
