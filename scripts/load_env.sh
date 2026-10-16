# Sourced by pixi on activation: export each KEY=value line of <repo>/.env, then the
# storage/ defaults. A variable already set in the environment is never overridden.
_reki_setdefault() {
    if [ -z "$(printenv "$1")" ]; then export "$1=$2"; fi
}
if [ -f "$PIXI_PROJECT_ROOT/.env" ]; then
    _n=0
    while IFS= read -r _line || [ -n "$_line" ]; do
        _n=$((_n + 1))
        _line=${_line%"$(printf '\r')"}
        case "${_line#"${_line%%[! 	]*}"}" in '' | \#*) continue ;; esac
        _line=${_line#export }
        _key=${_line%%=*}
        case "$_line" in
            *=*) case "$_key" in '' | [0-9]* | *[!A-Za-z0-9_]*) _key= ;; esac ;;
            *) _key= ;;
        esac
        if [ -z "$_key" ]; then
            printf 'load_env.sh: skipping malformed line %s of .env\n' "$_n" >&2
            continue
        fi
        _reki_setdefault "$_key" "${_line#*=}"
    done < "$PIXI_PROJECT_ROOT/.env"
fi
_reki_setdefault REKI_DATASETS_ROOT "$PIXI_PROJECT_ROOT/storage/datasets"
_reki_setdefault REKI_RUNS_DIR "$PIXI_PROJECT_ROOT/storage/runs"
_reki_setdefault REKI_SIM_ROOT "$PIXI_PROJECT_ROOT/storage/sim"
unset -f _reki_setdefault
unset _n _line _key
