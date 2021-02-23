#!/bin/bash
rm -f temp
ls *_1.sli | awk '{temp=$1; sub(/_1/,"_0",temp); print "mv",$1,temp;}' > temp
source temp

rm -f temp
ls *_2.sli | awk '{temp=$1; sub(/_2/,"_1",temp); print "mv",$1,temp;}' > temp
source temp

rm -f temp
ls *_3.sli | awk '{temp=$1; sub(/_3/,"_2",temp); print "mv",$1,temp;}' > temp
source temp

rm -f temp
ls *_4.sli | awk '{temp=$1; sub(/_4/,"_3",temp); print "mv",$1,temp;}' > temp
source temp
rm -f temp
