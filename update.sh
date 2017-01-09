#!/bin/sh
swift package update
swift package generate-xcodeproj
git apply generate_xcodeproj_fix.patch 
