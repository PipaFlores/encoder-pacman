import torch
import torch.nn as nn
## LSTM - AutoEncoder
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout = 0, seq_length=None):
        """
        Encoder LSTM network. It is a pretty standard LSTM implementation. 
        Only difference is that instead of outputting a next step prediction"""
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)

    def forward(self, X: torch.Tensor):
        out, (h, c) = self.lstm(X) # output hidden_state vector for each timestep, (last h, last c)
        x_encoding = h.squeeze(dim=0) #  [1, batch_size, hidden_size] -> [batch_size, hidden_size]
        ## Implementation from Matan Levi repeats the hidden vector to the seq_length here, but I will do it at the AE forward step.
        return x_encoding, out, (h, c)

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout= 0, seq_length=None, last_act = None, forced_teacher = False):
        """
        Decoder LSTM

        Args:
            input_size: number of dimensions of a timestep. If teacher forcing, is equal to data's dimensionality 
            and encoder's input size. Otherwise, it is the hidden_size (if feeding the repeated encoded data)
            hidden_size: size of hidden states (equal to encoder hidden_size).
            output_size: dimensionality of original data, to project the reconstruction from the hidden_size

        """
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.seq_length = seq_length

        self.last_act = last_act

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)

        self.fully_connected = nn.Linear(in_features=hidden_size, out_features = output_size)
    
    def forward(self, z: torch.Tensor, HC: tuple[torch.Tensor, torch.Tensor]):

        dec_output, (h, c) = self.lstm(z, HC) 

        if self.last_act:
            reconstruction = self.last_act(self.fully_connected(dec_output))
        else:
            reconstruction = self.fully_connected(dec_output)
        
        return reconstruction

class AELSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout= 0,last_act = None ,seq_length=None, forced_teacher = False):
        super(AELSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.seq_length = seq_length

        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size, dropout=dropout, seq_length=seq_length)

        if forced_teacher:
            self.decoder = Decoder(input_size=input_size, 
                                   hidden_size=hidden_size, 
                                   output_size=input_size, 
                                   dropout=dropout, 
                                   seq_length=seq_length)
        else:
            self.decoder = Decoder(input_size=hidden_size, 
                                   hidden_size=hidden_size, 
                                   output_size=input_size, 
                                   dropout=dropout, 
                                   last_act=last_act,
                                   seq_length=seq_length, 
                                   forced_teacher=forced_teacher)
        
        
    def forward(self, X: torch.Tensor, return_encoding= False):

        x_encoding, _, (h, c)= self.encoder(X)
        z = x_encoding.unsqueeze(1).repeat(1, X.shape[1] , 1) # [batch_dize, hidden_size] -> [batch_size, seq_length, hidden_size]
        reconstruction = self.decoder(z = z, HC = [h, c] )

        if return_encoding:
            return reconstruction, x_encoding
        return reconstruction
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr= 0.001)
    
    def loss(self, x_h, x):
        """
        Reconstruction MSE loss
        """
        return nn.functional.mse_loss(x_h, x, reduction="sum")

class UCR_Dataset(torch.utils.data.Dataset):

    def __init__(self, ucr_dataset):
        self.time_series = torch.Tensor(ucr_dataset[0]).transpose(1,2) # [N, channels, seq_length] -> [N, seq_length, channels]
        self.labels = ucr_dataset[1]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {
            "data": self.time_series[idx],
            "labels": self.labels[idx]
        }

class Trainer():
    def __init__(self, max_epochs= 50, batch_size= 32, val_set = True, gradient_clipping = None, verbose = True):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.val_set = val_set
        self.gradient_clipping = gradient_clipping
        self.verbose = verbose
        
    def fit(self, model:nn.Module , data: torch.utils.data.Dataset):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        optimizer = (
            self.model.configure_optimizer()
            if hasattr(self.model, "configure_optimizer")
            else torch.optim.Adam(self.model.parameters(), lr=0.001)
        )
        loss = self.model.loss if hasattr(self.model, "loss") and callable(self.model.loss) else lambda x_h, x: nn.MSELoss(reduction="sum")(x_h, x)

        if self.val_set:
            train_set, val_set = torch.utils.data.random_split(data, [0.7, 0.3])
            val_iter = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        else:
            train_set = data
        train_iter = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=False)

        self.train_loss_list, self.val_loss_list = [], []

        for epoch in range(self.max_epochs):
            self.model.train()
            loss_sum = 0

            for batch in train_iter:
                x = batch["data"].to(self.device)
                x_h = self.model(x)

                optimizer.zero_grad()
                batch_loss = loss(x_h, x)
                loss_sum += batch_loss.item()
                
                batch_loss.backward()
                # Gradient clipping in case of exploding gradients
                if self.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clipping)
                optimizer.step()

            if self.val_set:
                self.model.eval()
                val_loss_sum = 0 

                for batch in val_iter:
                    with torch.no_grad():
                        x = batch["data"].to(self.device)
                        x_h = self.model(x)

                        batch_loss = loss(x_h, x)
                        val_loss_sum += batch_loss.item()


            epoch_train_loss = loss_sum / len(train_iter.dataset)
            self.train_loss_list.append(epoch_train_loss)

            if self.val_set:
                epoch_val_loss = val_loss_sum / len(val_iter.dataset)
                self.val_loss_list.append(epoch_val_loss)

            if self.verbose:
                print(f"Epoch {epoch + 1}: Train loss={epoch_train_loss}, Val loss={epoch_val_loss if self.val_set else ''}")

    def plot_loss(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.plot(range(len(self.train_loss_list)), self.train_loss_list, label="Train Loss")
        if self.val_loss_list:
            ax.plot(range(len(self.val_loss_list)), self.val_loss_list, label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss")
        ax.legend()
                
