
for trainsample in 4000
do

for NUM_LAYER in 2
do

for LSTM_DIM in 200 
do

for dropout in 0.3 0.4 0.5
do

for normal in 1
do

for lexiconepoch in 0
do

for pretrain in giga
do


expname="SEMICRF_0920_1800_"$trainsample"_"$NUM_LAYER"_"$LSTM_DIM"_"$dropout"_"$normal"_"$lexiconepoch"_"$pretrain
echo $expname
echo "addr_$expname.log"
nohup python3 main.py --trial 0 --trainsample $trainsample --check_every 2000 --epoch 120 --NUM_LAYER $NUM_LAYER --LSTM_DIM $LSTM_DIM --WORD_DIM 100 --pretrain $pretrain --dropout $dropout --expname $expname --model CRF --normalize $normal --lexiconepoch $lexiconepoch > addr_$expname.log 2>&1 &
echo ""

done
done
done
done
done
done
done
