# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
#from chakraSimInput import Operator as op

import torch
from torch import nn
import torch.nn.functional as F
from torch.profiler import ExecutionTraceObserver, profile
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def check_device():
    print(f"PyTorch version:{torch.__version__}")  # 1.12.1 이상
    print(f"gpu support build: {torch.cuda.is_available()}")  # True 여야 합니다.
    

def trace_handler(prof):
    prof.export_chrome_trace("./log/kineto_trace.json")
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    check_device()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = Net()
    input = torch.randn(1, 1, 28, 28, requires_grad=True)
    labels = torch.tensor([1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    et = ExecutionTraceObserver()
    et.register_callback("./log/chakra_et.json")
    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=10, active=1),
        on_trace_ready=trace_handler
    ) as prof:
        for step in range(20):
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            if step == 11:
                et.stop()
            if step == 10:
                et.start()
            prof.step()
        et.unregister_callback()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

