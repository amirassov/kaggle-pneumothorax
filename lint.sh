#!/bin/bash

: '
Использование:
`./lint.sh <команда> <путь к директории с кодом>`,
например `./lint.sh format .`

<путь к директории> с кодом по умолчанию равен текущей папке, так что можно запускать и так из корневной директории проекта:
`./lint.sh format`

Команды:
format - отформатировать весь код
check - проверить, что код проходит проверку линтером и форматером
yapf-diff - показать diff между текущим кодом и кодом после форматирования
unify-diff - показать diff между текущим кодом и кодом, где все двойные кавычки будут заменены на одиночные
flake8 - проверить код линтером и показать ошибки
install - установить в текущее окружение библиотеки для форматирования и линтинга
'

COMMAND=$1
FILEPATH=${2:-.}

echo ${FILEPATH}

case ${COMMAND} in
    check)
        yapf --diff --recursive ${FILEPATH} > /dev/null 2>&1
        YAPF_EXIT_CODE=$?
        unify --check-only --recursive ${FILEPATH} > /dev/null 2>&1
        UNIFY_EXIT_CODE=$?
        flake8 ${FILEPATH} > /dev/null 2>&1
        FLAKE8_EXIT_CODE=$?
        if [[ $YAPF_EXIT_CODE -eq 1 ]]; then
            echo 'yapf error'
        fi
        if [[ $UNIFY_EXIT_CODE -eq 1 ]]; then
            echo 'unify error'
        fi
        if [[ $FLAKE8_EXIT_CODE -eq 1 ]]; then
            echo 'flake8 error'
        fi
        exit $YAPF_EXIT_CODE || $UNIFY_EXIT_CODE || $FLAKE8_EXIT_CODE
        ;;
    yapf-diff)
        yapf --diff --recursive ${FILEPATH}
        ;;
    unify-diff)
        unify --recursive ${FILEPATH}
        ;;
    format)
        unify --in-place --recursive ${FILEPATH}
        yapf --in-place --recursive ${FILEPATH}
        ;;
    flake8)
        flake8 ${FILEPATH}
        ;;
    install)
        pip3 install yapf unify flake8
        ;;
    *)
        echo $"Usage: $0 {check|yapf-diff|unify-diff|format|flake8|install}"
        exit 1
esac
