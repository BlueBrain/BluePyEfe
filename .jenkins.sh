#!/bin/bash

set -e
set -x

git config --global http.proxy http://bbpproxy.epfl.ch:80/
git config --global https.proxy http://bbpproxy.epfl.ch:80/

tox_args='-v --recreate'

if [ "${os}" = "bb5" ]
then
	. /opt/rh/rh-python36/enable
    tox_args="${tox_args} -e py3-functional"
elif  [ "${os}" = "Ubuntu-18.04" ]
then
    tox_args="${tox_args} -e py3-functional-devpi"
else
    tox_args="${tox_args} -e py27-functional"
fi

cd $WORKSPACE

#########
# Virtualenv
#########

if [ ! -d "${WORKSPACE}/env" ]; then
  virtualenv ${WORKSPACE}/env --no-site-packages
fi

. ${WORKSPACE}/env/bin/activate
pip install pip --upgrade
pip install tox --upgrade   


#####
# Tests
#####

cd  ${WORKSPACE}/BluePyEfe

tox ${tox_args}
