: python run_model.py --model=baseline --do_train=true --do_eval=true --do_predict=true --train_batch_size=128 --data_dir=data/msr --vocab_file=data/vocab.txt --bigram_file=data/bigram.txt --config_file=config/baseline.json --output_dir=output/baseline/msr --init_embedding=data/wordvec_100 --learning_rate=0.001 --num_train_epochs=5.0 
: python run_model.py --model=baseline --do_train=true --do_eval=true --do_predict=true --train_batch_size=128 --data_dir=data/as --vocab_file=data/vocab.txt --bigram_file=data/bigram.txt --config_file=config/baseline.json --output_dir=output/baseline/as --init_embedding=data/wordvec_100 --learning_rate=0.001 --num_train_epochs=5.0 
: python run_model.py --model=baseline --do_train=true --do_eval=true --do_predict=true --train_batch_size=128 --data_dir=data/cityu --vocab_file=data/vocab.txt --bigram_file=data/bigram.txt --config_file=config/baseline.json --output_dir=output/baseline/cityu --init_embedding=data/wordvec_100 --learning_rate=0.001 --num_train_epochs=5.0 
: python run_model.py --model=baseline --do_train=true --do_eval=true --do_predict=true --train_batch_size=128 --data_dir=data/pku --vocab_file=data/vocab.txt --bigram_file=data/bigram.txt --config_file=config/baseline.json --output_dir=output/baseline/pku --init_embedding=data/wordvec_100 --learning_rate=0.001 --num_train_epochs=5.0 

python run_model.py --model=dict_concat --do_train=true --do_eval=true --do_predict=true --train_batch_size=128 --data_dir=data/msr --vocab_file=data/vocab.txt --bigram_file=data/bigram.txt --dict_file=data/dict/dict_1 --config_file=config/dict_concat.json --output_dir=output/dict_concat/msr --init_embedding=data/wordvec_100 --learning_rate=0.001 --num_train_epochs=5.0 
python run_model.py --model=dict_concat --do_train=true --do_eval=true --do_predict=true --train_batch_size=128 --data_dir=data/as --vocab_file=data/vocab.txt --bigram_file=data/bigram.txt --dict_file=data/dict/dict_2 --config_file=config/dict_concat.json --output_dir=output/dict_concat/as --init_embedding=data/wordvec_100 --learning_rate=0.001 --num_train_epochs=5.0 
python run_model.py --model=dict_concat --do_train=true --do_eval=true --do_predict=true --train_batch_size=128 --data_dir=data/cityu --vocab_file=data/vocab.txt --bigram_file=data/bigram.txt --dict_file=data/dict/dict_2 --config_file=config/dict_concat.json --output_dir=output/dict_concat/cityu --init_embedding=data/wordvec_100 --learning_rate=0.001 --num_train_epochs=5.0 
python run_model.py --model=dict_concat --do_train=true --do_eval=true --do_predict=true --train_batch_size=64 --data_dir=data/pku --vocab_file=data/vocab.txt --bigram_file=data/bigram.txt --dict_file=data/dict/dict_1 --config_file=config/dict_concat.json --output_dir=output/dict_concat/pku --init_embedding=data/wordvec_100 --learning_rate=0.001 --num_train_epochs=5.0 
pause