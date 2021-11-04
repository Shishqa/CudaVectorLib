block_sizes=(
  128
  256
  512
  1024
)

arr_sizes=(
  1000
  10000
  100000
  1000000
  33554432
  67108864
  134217728
  268435450
)

echo 'arr,block,time'
for a in ${arr_sizes[@]}
do
  for b in ${block_sizes[@]}
  do
    ./$@ $a $b
  done
done
