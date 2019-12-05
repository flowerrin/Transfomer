import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, batch_size=96):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        
    def forward(self, encoder_input, decoder_input, flag=1):
        encoder_output = self.encoder(encoder_input, flag)
        decoder_output = self.decoder(decoder_input, encoder_output, flag, encoder_input)
        return decoder_output
    
    
    def encode(self, encoder_input, flag=1):
        return self.encoder(encoder_input, flag)
    
    def decode(self, decoder_input, encoder_output, encoder_input, flag=1):
        return self.decoder(decoder_input, encoder_output, flag, encoder_input)
