model=$1
for trainsample in -1
do

for NUM_LAYER in 2
do

for LSTM_DIM in 200
do

for dropout in 0
do

for normal in 1
do

for lexiconepoch in 0
do

for pretrain in giga
do

for singleton in 0
do

for syntacticcomposition in 1
do

expname=$model"_0901_0800_"$trainsample"_"$NUM_LAYER"_"$LSTM_DIM"_"$dropout"_"$normal"_"$lexiconepoch"_"$syntacticcomposition"_"$singleton"_"$pretrain
echo $expname
echo "addr_$expname.log"
nohup python3 main.py --trial 0 --trainsample $trainsample --check_every 1000 --epoch 15 --NUM_LAYER $NUM_LAYER --LSTM_DIM $LSTM_DIM --WORD_DIM 100 --pretrain $pretrain --dropout $dropout --expname $expname --model $model --normalize $normal --lexiconepoch $lexiconepoch --syntactic_composition $syntacticcomposition --singleton $singleton > addr_$expname.log 2>&1 &
echo ""

done
done
done
done
done
done
done
done
done
