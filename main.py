
from allocation.resource import CloudServer

server = CloudServer(
    server_id="Server-1",
    total_cpu=8,
    total_ram=16
)

print("Server:", server.server_id)
print("Available CPU:", server.available_cpu())
print("Available RAM:", server.available_ram(), "GB")
