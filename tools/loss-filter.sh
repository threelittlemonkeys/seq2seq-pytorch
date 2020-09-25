if [ $# -ne 7 ]
then
    echo "Usage: $0 script_path raw_data tkn_data output src_lang tgt_lang id"
    exit
fi

set -x
fn_raw=$2 # raw corpus
fn_tkn=$3 # tokenized corpus
fn_out=$4 # output
src=$5 # source language
tgt=$6 # target language
num_epochs=5 # number of epochs
batch_size=100000 # batch size
id=$(printf "%03d" $7) # identifier
ln=$((($7 - 1) * batch_size + 1)) # starting line number
script_path=$1 # seq2seq script path

if [ ${script_path: -1} == "/" ]
then
    script_path=${script_path:0:-1}
fi

sed -n "$ln,$((ln + batch_size - 1))p" $fn_raw > $fn_out.$id.raw
sed -n "$ln,$((ln + batch_size - 1))p" $fn_tkn > $fn_out.$id
# cat | tail -n +$ln | head -$batch_size
cut -f1 $fn_out.$id > $fn_out.$id.$src
cut -f2 $fn_out.$id > $fn_out.$id.$tgt
paste $fn_out.$id.$src $fn_out.$id.$tgt > $fn_out.$id.$src$tgt
paste $fn_out.$id.$tgt $fn_out.$id.$src > $fn_out.$id.$tgt$src

fn=$fn_out.$id.$src$tgt
python3 $script_path/prepare.py $fn
python3 $script_path/train.py $fn.model $fn.src.char_to_idx $fn.src.word_to_idx $fn.tgt.word_to_idx $fn.csv $num_epochs

fn=$fn_out.$id.$tgt$src
python3 $script_path/prepare.py $fn
python3 $script_path/train.py $fn.model $fn.src.char_to_idx $fn.src.word_to_idx $fn.tgt.word_to_idx $fn.csv $num_epochs

fn=$fn_out.$id
python3 $script_path/loss-filter.py $fn.raw $fn $fn.$src$tgt.idx $fn.$src$tgt.model.epoch$num_epochs.loss $fn.$tgt$src.idx $fn.$tgt$src.model.epoch5.loss > $fn.dcce.tsv

zip $fn.dcce.data.zip $fn.raw $fn $fn.$src$tgt.idx $fn.$src$tgt.model.epoch$num_epochs.loss $fn.$tgt$src.idx $fn.$tgt$src.model.epoch5.loss

rm $fn
rm $fn.raw
rm $fn.$src*
rm $fn.$tgt*
