#!/bin/bash

spack load awscli

cd /project/dane2/wficai/pda/external_validation/Boston/exports

aws s3 sync . s3://fast-videos/pda/external_validation/boston/
