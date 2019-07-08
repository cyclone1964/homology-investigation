# This simple awk script parses the output of ripser and prints out just the
# bar codes as lines with start, end, and dimension on them
/persistence/ {
    sub(/:/,"")
    dimension = $5
}
(NF == 1) {
    gsub(/[\[,\)]/," ")
    print $1,$2,dimension
}
