from utils import CharacterTable, transform2
from utils import batch, datagen, datagen_simple

def load_files(data_dir,fold_num):  
    # extract training tokens
    # train_dec_tokens are targets and train_tokens inputs
    with open(data_dir+"/foldset"+str(fold_num)+"/train") as f:
        train_tups = [line.strip('\n').split(',') for line in f]
    train_dec_tokens, train_tokens = zip(*train_tups)

    # Convert train word token lists to type lists
    #train_tokens,train_dec_tokens = get_type_lists(train_tokens,train_dec_tokens)

    # extract validation tokens
    # val_dec_tokens are targets and val_tokens inputs
    with open(data_dir+"/foldset"+str(fold_num)+"/val") as f:
        val_tups = [line.strip('\n').split(',') for line in f]
    val_dec_tokens, val_tokens = zip(*val_tups)

    input_chars = set(' '.join(train_tokens) + '*' + '\t') # * and \t are EOS and SOS respectively
    target_chars = set(' '.join(train_dec_tokens) + '*' + '\t')
    total_chars = input_chars.union(target_chars)
    #nb_input_chars = len(input_chars)
    #nb_target_chars = len(target_chars)
    nb_total_chars = len(total_chars)
    # Define training and evaluation configuration.
    #input_ctable  = CharacterTable(input_chars)
    #target_ctable = CharacterTable(target_chars)
    total_ctable = input_ctable = target_ctable = CharacterTable(total_chars)

    maxlen = max([len(token) for token in train_tokens]) + 2

    return train_tokens, train_dec_tokens, val_tokens, val_dec_tokens, input_ctable, target_ctable, nb_total_chars, total_ctable, maxlen

def get_iter(epoch, num_epochs, data_dir, fold_num=1, train_batch_size=128, val_batch_size=256,  predict=False):
    reverse = False

    train_tokens, train_dec_tokens, val_tokens, val_dec_tokens , input_ctable, target_ctable, nb_total_chars, total_ctable, maxlen = load_files(data_dir=data_dir,fold_num=fold_num)

    if predict == False:
        #st_ind,et_ind = int(len(train_tokens) * (epoch/num_epochs)), int(len(train_tokens)*((epoch+1)/num_epochs))
        #sv_ind,ev_ind = int(len(val_tokens) * (epoch/num_epochs)), int(len(val_tokens)*((epoch+1)/num_epochs))
        st_ind,et_ind = int(train_batch_size*(epoch/num_epochs)), int(train_batch_size*((epoch+1)/num_epochs) )
        sv_ind,ev_ind = int(val_batch_size*(epoch/num_epochs)), int(val_batch_size*((epoch+1)/num_epochs))

    else:
        st_ind,et_ind = int(train_batch_size*(epoch/num_epochs)), int(train_batch_size*((epoch+1)/num_epochs) )
        sv_ind,ev_ind = int(val_batch_size*(epoch/num_epochs)), int(val_batch_size*((epoch+1)/num_epochs))
   
    train_encoder, train_decoder, train_target = transform2( train_tokens[st_ind:et_ind], maxlen, shuffle=False, dec_tokens=train_dec_tokens[st_ind:et_ind])
    val_encoder, val_decoder, val_target = transform2( val_tokens[sv_ind:ev_ind], maxlen, shuffle=False, dec_tokens=val_dec_tokens[sv_ind:ev_ind])

    train_batch_size, val_batch_size = len(train_encoder), len(val_encoder)
    train_encoder_batch = batch(train_encoder, maxlen, input_ctable, train_batch_size, reverse)
    train_decoder_batch = batch(train_decoder, maxlen, target_ctable, train_batch_size)
    train_target_batch  = batch(train_target, maxlen, target_ctable, train_batch_size)

    val_encoder_batch = batch(val_encoder, maxlen, input_ctable, val_batch_size, reverse)
    val_decoder_batch = batch(val_decoder, maxlen, target_ctable, val_batch_size)
    val_target_batch  = batch(val_target, maxlen, target_ctable, val_batch_size)

    train_loader = datagen_simple(train_encoder_batch, train_target_batch)
    val_loader = datagen_simple(val_encoder_batch, val_target_batch)
    
    if predict == True: return train_loader, val_loader, total_ctable
    return train_loader, val_loader, train_tokens, val_tokens, nb_total_chars, total_ctable
