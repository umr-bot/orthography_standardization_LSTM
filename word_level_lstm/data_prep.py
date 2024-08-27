SOS, EOS = '\t', '*'
def transform2(tokens, maxlen, shuffle=False, dec_tokens=[], chrs=[], reverse=False):
    """Transform tokens into model inputs and targets.
    All inputs and targets are padded to maxlen with EOS character.
    """
    encoder_tokens = []
    decoder_tokens = []
    target_tokens = []

    assert(len(tokens)==len(dec_tokens))
    for i in range(len(tokens)):
        token,dec_token = tokens[i], dec_tokens[i]
        if len(token) > 0: # only deal with tokens longer than length 3
            #encoder = add_speling_erors(token, error_rate=error_rate)
            encoder = token
            encoder += EOS * (maxlen - len(encoder)) # Padded to maxlen.
            if reverse: encoder = encoder[::-1]
            #encoder_tokens.append(encoder)
        
            decoder = SOS + dec_token
            decoder += EOS * (maxlen - len(decoder))
            #decoder_tokens.append(decoder)
        
            target = decoder[1:]
            target += EOS * (maxlen - len(target))
            #target_tokens.append(target)
            if (len(encoder) == len(decoder) == len(target)):
                encoder_tokens.append(encoder)
                decoder_tokens.append(decoder)
                target_tokens.append(target)
            else: continue

    return encoder_tokens, decoder_tokens, target_tokens

def datagen_simple(input_iter, target_iter):
    """Utility function to load data into required model format."""
    while(True):
        input_ = next(input_iter)
        target = next(target_iter)
        yield (input_, target)

