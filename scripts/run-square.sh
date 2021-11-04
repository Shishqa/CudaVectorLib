block_sizes=(
  4
  8
  16
  32
)

arr_sizes=(
  128
  256
  512
  1024
  2048
  4096
  8192
  16384
)

echo 'arr,block,time'
for a in ${arr_sizes[@]}
do
  for b in ${block_sizes[@]}
  do
    ./$@ $a $b
  done
done
