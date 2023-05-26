# instructpix2pixcogcrea
Repository using Cog and a Caddy reverse-proxy to use a cog container to run several predictions at the same time.

## Structure of the localhost
To run predictions: 
:5000/predictions

To retrieve the results: 
:8000/<index>.png
  
## Installation with different service providers

### Lambda labs
When the instance on lambda labs is created and the SSH key setup run on your computer:
```console
ssh -i <your-ssh-key> ubuntu@<IP> 
```
And then on the machine:
```console
git clone https://github.com/brabecq/instructpix2pixcogcrea.git

cd instructpix2pixcogcrea
chmod +x makefile.sh
./makefile.sh
```

### Paperspace
```console
ssh paperspace@<IP>
```
And then on the machine:
```console
git clone https://github.com/brabecq/instructpix2pixcogcrea.git

cd instructpix2pixcogcrea
chmod +x makefile_paperspace.sh
./makefile_paperspace.sh
```
Reboot the machine and:

```console
cd instructpix2pixcogcrea
chmod +x makefile.sh
./makefile.sh
```
