import carla
import getpass
import os
import socket
import sys

print("python=", sys.executable)
print("user=", getpass.getuser())
print("username_env=", os.environ.get("USERNAME"))
print("userprofile=", os.environ.get("USERPROFILE"))

s = socket.create_connection(("127.0.0.1", 2000), 5)
print("tcp_ok")
s.close()

c = carla.Client("127.0.0.1", 2000)
c.set_timeout(10.0)
print("server=", c.get_server_version())
print("map=", c.get_world().get_map().name)
