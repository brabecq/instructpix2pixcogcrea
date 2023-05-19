# instructpix2pixcogcrea
Cog repo for instructpix2pix for the crea.visions project

## Lambda labs
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

## Paperspace
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
git clone https://github.com/brabecq/instructpix2pixcogcrea.git

cd instructpix2pixcogcrea
chmod +x makefile.sh
./makefile.sh
```

And that's it!
