#!/bin/bash

tritonserver --model-repository=/models &

quick-deploy $@