# phageid

A command-line toolfor automating phage plaque counting.

## installation instructions

The whole project installs with pip:

```sh
pip git+https://github.com/FinleyGibson/phageid
```

## Usage

The project has two functions the first is to split the individual trays out of the original images:

```sh
phageid path/to/image/dir/ output/dir/
```

and then to detect the phage plaques in the .npy files produced:

```sh
phageid detect path/to/input.npy output/dir/ --visualise
```

each has the optional `--visualise` flag. Which Shows some of the process. It is recommended to use this flag initually, until you are happy that the process is working as intended.

## Configuration

On first running, the program generates a configuration file at the following locations

Mac/Linux:

```sh
~/.config/phageid/config.toml
```

Windows:

```sh
C:\Users\<YourUsername>\AppData\Roaming\phageid\config.toml
```
