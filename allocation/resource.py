
class CloudServer:
    def __init__(self, server_id, total_cpu, total_ram):
        self.server_id = server_id
        self.total_cpu = total_cpu
        self.total_ram = total_ram

        self.used_cpu = 0
        self.used_ram = 0

    def available_cpu(self):
        return self.total_cpu - self.used_cpu

    def available_ram(self):
        return self.total_ram - self.used_ram
