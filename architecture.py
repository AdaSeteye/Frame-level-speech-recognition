from imports import *
import config


class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 3840),
            nn.LayerNorm(3840),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(3840, 1792),
            nn.LayerNorm(1792),
            nn.SiLU(),
            nn.Dropout(0.25),

            nn.Linear(1792, 1280),
            nn.LayerNorm(1280),
            nn.SiLU(),
            nn.Dropout(0.2),

            nn.Linear(1280, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(256, output_size)
        )




        if config['weight_initialization'] is not None:
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                if config["weight_initialization"] == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                elif config["weight_initialization"] == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                elif config["weight_initialization"] == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif config["weight_initialization"] == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif config["weight_initialization"] == "uniform":
                    torch.nn.init.uniform_(m.weight)
                else:
                    raise ValueError("Invalid weight_initialization value")

                m.bias.data.fill_(0)


    def forward(self, x):

        x = torch.flatten(x, start_dim=1)  

        return self.model(x)