import numpy as np
import math
import torch
import torch.nn as nn

import random


# helper functions
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return torch.from_numpy(pos_encoding).type(torch.FloatTensor)


# base Seq2Seq Model (LSTM Encoder - LSTM Decoder)
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim
        )

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src):
        # src = [src sent len, batch size]

        # Compute an embedding from the src data and apply dropout to it
        # embedded = [src sent len, batch size, emb_dim]
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)

        # Compute the RNN output values of the encoder RNN.
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
        # outputs = [src sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # outputs are always from the top hidden layer
        output, (hidden, cell) = self.rnn(embedded)
        return output, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout_p):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout_p
        )

        self.out = nn.Linear(
            in_features=hid_dim,
            out_features=output_dim
        )

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]
        # input = [1, batch size]
        input = input.unsqueeze(0)

        # Compute an embedding from the input data and apply dropout to it
        # embedded = [1, batch size, emb dim]
        embedded = self.dropout(self.embedding(input))

        # Compute the RNN output values of the encoder RNN.
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)

        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # prediction = [batch size, output dim]
        prediction = self.out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # tgt = [tgt sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        # Again, now batch is the first dimension instead of zero
        batch_size = tgt.shape[1]
        max_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, tgt_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        _, hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = tgt[0, :]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (tgt[t] if teacher_force else top1)

        return outputs


# Seq2Seq Model with multiple layers CNN Encoder and LSTM Decoder
class EncoderCNN(nn.Module):
    def __init__(self, device, input_dim, emb_dim, hid_dim, num_layers=5,
                 out_dim=None, pos_encoding=False,
                 dropout_p=0.2, kernel_size=3):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.pos_encoding = pos_encoding
        self.kernel_size = kernel_size
        self.device = device

        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim,
        )

        self.conv = nn.ModuleList([nn.Conv1d(in_channels=emb_dim,
                                             out_channels=hid_dim,
                                             kernel_size=kernel_size,
                                             padding=kernel_size // 2) for _ in range(num_layers)])

        # # the size of the output from all convolutions and max pooling is [num_kernels, hid_dim]
        # output_shape = len(kernel_sizes) * self.hid_dim
        if out_dim:
            self.projection_layer = nn.Linear(hid_dim, out_dim)
        else:
            self.projection_layer = None
            self.out_dim = hid_dim

        self.dropout = nn.Dropout(p=dropout_p)

        self.pooling = nn.AdaptiveMaxPool1d(output_size=1)

        self.relu = nn.ReLU()

    def forward(self, src):
        # src = [src_sent_len, batch_size]
        # self.embedding => [src_sent_len, batch_size, emb_dim]
        # permute => [batch_size, emb_dim, src sent len] - what we need for conv layer
        embedded = self.embedding(src)

        if self.pos_encoding:
            pos_encoding_emb = torch.zeros([src.shape[1], src.shape[0], self.emb_dim]).type(torch.FloatTensor).to(
                self.device)
            pos_encoding_emb[:, :, :] = positional_encoding(src.shape[0], self.emb_dim)
            embedded += pos_encoding_emb.permute((1, 0, 2))

        embedded = embedded.permute((1, 2, 0))
        embedded = self.dropout(embedded)

        cnn_output = embedded
        for i, conv_layer in enumerate(self.conv):
            cnn_output = self.relu(conv_layer(cnn_output) + cnn_output)

        encoder_output = torch.flatten(self.pooling(cnn_output), start_dim=1)

        if self.projection_layer:
            encoder_output = self.projection_layer(encoder_output)

        return encoder_output


class CNN2RNN(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # tgt = [tgt sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        # Again, now batch is the first dimension instead of zero
        batch_size = tgt.shape[1]
        max_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, tgt_vocab_size).to(self.device)

        enc_output = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = tgt[0, :]

        # processing the first input to the RNN decoder from the CNN encoder
        dec_hid_size = self.decoder.hid_dim
        output, hidden, cell = self.decoder(input,
                                            enc_output[:, :2 * dec_hid_size].contiguous().view(2, batch_size,
                                                                                               dec_hid_size),
                                            enc_output[:, 2 * dec_hid_size:].contiguous().view(2, batch_size,
                                                                                               dec_hid_size))
        outputs[1] = output
        teacher_force = random.random() < teacher_forcing_ratio
        top1 = output.max(1)[1]
        input = (tgt[1] if teacher_force else top1)

        for t in range(2, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (tgt[t] if teacher_force else top1)

        return outputs


# Seq2Seq model with LSTM Encoder - LSTM Decoder with implementation of Attention Mechanism
class Attention(nn.Module):

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                over which to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights


class DecoderAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout_p=0.5, attention_type="general"):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout_p
        )

        self.attention = Attention(hid_dim, attention_type=attention_type)

        self.out = nn.Linear(
            in_features=hid_dim,
            out_features=output_dim
        )

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, input, encoder_context, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]
        # input = [1, batch size]
        input = input.unsqueeze(0)

        # Compute an embedding from the input data and apply dropout to it
        # embedded = [1, batch size, emb dim]
        embedded = self.dropout(self.embedding(input))

        # Compute the RNN output values of the encoder RNN.
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)

        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        attention_output, _ = self.attention(output.transpose(0, 1), encoder_context.transpose(0, 1))
        attention_output = attention_output.transpose(0, 1)

        # prediction = [batch size, output dim]
        prediction = self.out(attention_output.squeeze(0))
        return prediction, hidden, cell


class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert isinstance(self.decoder, DecoderAttention), \
            "Decoder must be an instance of DecoderAttention class!"

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # tgt = [tgt sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        # Again, now batch is the first dimension instead of zero
        batch_size = tgt.shape[1]
        max_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, tgt_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_context, hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = tgt[0, :]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, encoder_context, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (tgt[t] if teacher_force else top1)

        return outputs


# Implementation of Transformer architecture using nn.Transformer class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, src):
        src = src + self.scale * self.pe[:src.size(0), :]
        return self.dropout(src)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_dim, tgt_dim, emb_dim, enc_layers=1, dec_layers=1,
                 n_heads=1, dim_feedforward=2048, dropout=0.1, activation="relu",
                 pad_idx=1, sos_idx=2, device="cuda"):
        super(Seq2SeqTransformer, self).__init__()

        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.emb_dim = emb_dim

        self.enc_emb = nn.Embedding(src_dim, emb_dim)
        self.dec_emb = nn.Embedding(tgt_dim, emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim, dropout)

        self.transformer_model = nn.Transformer(d_model=emb_dim,
                                                nhead=n_heads,
                                                num_encoder_layers=enc_layers,
                                                num_decoder_layers=dec_layers,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                activation=activation)

        self.linear = nn.Linear(emb_dim, tgt_dim)

        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.device = device

        self.src_mask = None
        self.tgt_mask = None
        self.memory_mask = None

    def make_len_mask(self, inp):
        return inp == self.pad_idx

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        output_len = tgt.shape[0]
        batch_size = tgt.shape[1]
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        src_key_padding_mask = self.make_len_mask(src)
        # tgt_key_padding_mask = self.make_len_mask(tgt)

        src = (self.enc_emb(src) * math.sqrt(self.emb_dim)).transpose(0, 1)
        src = self.pos_encoder(src)

        if self.training:
            start_idxes = torch.ones([batch_size, 1], dtype=torch.int64, device=self.device) * self.sos_idx

            # right shift decoder input by adding start symbol
            tgt = torch.cat((start_idxes, tgt), 1)[:, :-1]

            tgt = (self.dec_emb(tgt) * math.sqrt(self.emb_dim)).transpose(0, 1)
            tgt = self.pos_encoder(tgt)

            tgt_mask = self.transformer_model.generate_square_subsequent_mask(tgt.size(0)).to(self.device)

            decoder_outputs = self.transformer_model(
                src=src,
                tgt=tgt,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask)

            decoder_outputs = self.linear(decoder_outputs)

        else:
            encoder_hidden_states = self.transformer_model.encoder(src, src_key_padding_mask=src_key_padding_mask)

            decoder_inputs = torch.empty((src.size(1), output_len + 1),
                                         dtype=torch.int64, device=self.device).fill_(self.sos_idx)

            decoder_outputs = torch.zeros(output_len, src.size(1), self.tgt_dim, device=self.device)

            for i in range(output_len):
                decoder_input = (self.dec_emb(decoder_inputs[:, :i + 1]) * math.sqrt(self.emb_dim)).transpose(0, 1)
                decoder_input = self.pos_encoder(decoder_input)

                tgt_mask = self.transformer_model.generate_square_subsequent_mask(i + 1).to(self.device)

                decoder_output = self.transformer_model.decoder(
                    tgt=decoder_input,
                    memory=encoder_hidden_states,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_key_padding_mask)

                decoder_output = self.linear(decoder_output)[-1]

                decoder_outputs[i] = decoder_output

                decoder_inputs[:, i + 1] = decoder_output.max(1)[1]

        return decoder_outputs


# Another implementation (it looks like there should be a difference in training and evaluation behaviour)
class LanguageTransformer(nn.Module):
    def __init__(self, src_dim, trg_dim, emb_dim, num_encoder_layers, num_decoder_layers, nhead, dim_feedforward=2048,
                 pos_dropout=0.1, trans_dropout=0.1, device="cuda", pad_idx=1):
        super(LanguageTransformer, self).__init__()
        self.emb_dim = emb_dim

        self.embed_src = nn.Embedding(src_dim, emb_dim)
        self.embed_tgt = nn.Embedding(trg_dim, emb_dim)

        self.pos_enc = PositionalEncoding(emb_dim, pos_dropout)

        self.transformer = nn.Transformer(emb_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                                          trans_dropout)

        self.linear = nn.Linear(emb_dim, trg_dim)

        self.device = device
        self.pad_idx = pad_idx

    def make_key_pad_mask(self, inp):
        return (inp == self.pad_idx).transpose(0, 1)

    def forward(self, src, tgt):
        # src = [src sent len, batch size]
        # tgt = [tgt sent len, batch size]

        src_key_pad_mask = self.make_key_pad_mask(src).to(self.device)
        tgt_key_pad_mask = self.make_key_pad_mask(tgt).to(self.device)
        memory_key_pad_mask = src_key_pad_mask.clone()

        tgt_inp, tgt_out = tgt[:-1, :], tgt[1:, :]
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_inp.shape[0]).to(self.device)

        src = self.pos_enc(self.embed_src(src) * math.sqrt(self.emb_dim))
        tgt_inp = self.pos_enc(self.embed_tgt(tgt_inp) * math.sqrt(self.emb_dim))

        output = self.transformer(src, tgt_inp, tgt_mask=tgt_mask, src_key_padding_mask=src_key_pad_mask,
                                  tgt_key_padding_mask=tgt_key_pad_mask[:, :-1],
                                  memory_key_padding_mask=memory_key_pad_mask)

        return self.linear(output)