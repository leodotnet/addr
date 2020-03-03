
for trainsample in -1
do

for NUM_LAYER in 2
do

for LSTM_DIM in 200 
do

for dropout in 0 0.1
do

for normal in 1
do

for lexiconepoch in 0
do

for pretrain in giga 
do

for syntacticcomposition in 2
do

for MAXLLIMIT in 13 8 #9 10 11 13
do

expname="SEMICRF_1120_1500_"$trainsample"_"$NUM_LAYER"_"$LSTM_DIM"_"$dropout"_"$normal"_"$lexiconepoch"_"$pretrain"_"$syntacticcomposition"_"$MAXLLIMIT
echo $expname
echo "addr_$expname.log"
nohup python3 main.py --trial 0 --trainsample $trainsample --check_every 5000 --epoch 20 --NUM_LAYER $NUM_LAYER --LSTM_DIM $LSTM_DIM --WORD_DIM 100 --pretrain $pretrain --dropout $dropout --expname $expname --model SEMICRF --normalize $normal --lexiconepoch $lexiconepoch --syntactic_composition $syntacticcomposition --MAXLLIMIT $MAXLLIMIT > addr_$expname.log 2>&1 &
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
