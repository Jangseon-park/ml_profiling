import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torch.profiler import profile, record_function, ProfilerActivity, ExecutionTraceObserver
import torch.multiprocessing as mp
import torch.nn.functional as F
import os


def trace_handler(prof):
    pid = os.getpid()
    prof.export_chrome_trace("kineto_trace" + str(pid) + ".json")


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.drop2D = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.drop2D(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)


def train(rank, world_size):
    setup(rank, world_size)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=64)

    model = SimpleCNN().cuda(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    et = ExecutionTraceObserver()
    pid = os.getpid()
    et.register_callback("pytorch_et" + str(pid) + ".json")

    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            on_trace_ready=trace_handler,
            record_shapes=True,
            with_stack=True
    ) as prof:
        for epoch in range(1, 2):  # ?~D?~K??~U~\ ?~X~H?~K~\를 ?~\~D?~U? ?~W~P?~O??~A?를 1?~\ ?~D??| ~U
            model.train()
            sampler.set_epoch(epoch)
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.cuda(rank), target.cuda(rank)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx == 11:
                    et.stop()
                if batch_idx == 10:
                    et.start()
                if batch_idx % 10 == 0:
                    print(f'Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')
                prof.step()


def main():
    world_size = 2
    mp.spawn(train,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    main()

