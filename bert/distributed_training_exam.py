import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# 예제 데이터셋 정의
class ExampleDataset(Dataset):
    def __len__(self):
        return 1000  # 예제 데이터셋 크기

    def __getitem__(self, idx):
        return torch.tensor([idx], dtype=torch.float), torch.tensor([idx], dtype=torch.float)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 분산 환경 초기화
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size):
    setup(rank, world_size)

    # 데이터셋과 데이터로더 설정
    dataset = ExampleDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

    # 모델, 손실 함수, 옵티마이저 설정
    model = nn.Linear(1, 1).cuda(rank)
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # 학습 루프
    for epoch in range(10):
        for data, target in dataloader:
            optimizer.zero_grad()
            output = ddp_model(data.cuda(rank))
            loss = loss_fn(output, target.cuda(rank))
            loss.backward()
            optimizer.step()
        print(f'Rank {rank}, Epoch {epoch}, Loss {loss.item()}')

    cleanup()


def main():
    world_size = 2  # 사용할 GPU 수
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()