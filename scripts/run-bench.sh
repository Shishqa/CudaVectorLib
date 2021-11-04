linear=(
  01-add
  02-mul
  05-scalar-mul
  06-cosine-vector
)

square=(
  03-matrix-add
  07-matrix-mul
)

mkdir -p data

for p in ${linear[@]}
do
  ./scripts/run-linear.sh ./build/$p > data/$p.csv
done

for p in ${square[@]}
do
  ./scripts/run-square.sh ./build/$p > data/$p.csv
done
