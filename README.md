# honors_project_computersystems

## Server
Running the server is simple.
Python3 Federated_server.py

The following parameters can be applied:

--host The IP address of the server.

--port The port the server is using

--clients The expected amount of clients for the session.

--aggregation you can choose the type of aggregation, from fedavg to fedsgd

## Client
Running a client is simple.
Python3 federated_client.py --id #

The following parameters can be applied:

--id Sets the ID of the client connecting.

--host Points to the IP address of the server. Default is localhost.

--port Points to the port the server is using. Default is 5000.

--rounds Sets the amount of rounds the client will do. Please make sure all clients are doing an equal number of rounds.

--timesteps Sets the amount of timesteps performed per round before having the server gather and redistribute weights.
