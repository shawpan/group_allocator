# Setup

requires `python3.7` and `docker`

```
git clone https://github.com/shawpan/group_allocator.git
cd group_allocator
bash create_env.sh
source group_allocator_environment/bin/activate
bash setup.sh
```

it will install the required packages and prepare for starting

# Start the app

```
bash start_app.sh
```

it will run tensorflow_model_server in docker and api server
