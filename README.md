## Parameter Estimation Benchmark Importer
This project holds the code to import the parameter estimation benchmark problems into [COPASI](https://copasi.org). The test cases it ought to be able to deal with are from this project: 

* [Benchmark-Models](https://github.com/LeonardSchmiester/Benchmark-Models)

they are also added as submodule to this repo. So if you do want to use those, be sure to check it out as well, running: 

	git submodule init 
	git submodule update

### Setup
Create a new virtual environment, and then run `pip install requirements.txt`. Dependencies, are: 

* pyqt5
* python-copasi
* pandas 

### Usage
Once installed, you can use the graphical user interface, specify the benchmark directory, select the test and the model, and you ought to be able to open the generated COPASI file directly.

### License
Just as COPASI, the packages available on this page are provided under the 
[Artistic License 2.0](http://copasi.org/Download/License/), 
which is an [OSI](http://www.opensource.org/) approved license. This license 
allows non-commercial and commercial use free of charge.