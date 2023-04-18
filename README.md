## PEtab Importer
This project holds the code to import the parameter estimation benchmark problems into [COPASI](https://copasi.org). The test cases it ought to be able to deal with are from this project: 

* [Benchmark-Models](https://github.com/LeonardSchmiester/Benchmark-Models)

These models are encoded in the [PEtab](https://github.com/ICB-DCM/PEtab/) format with this [documentation](https://github.com/ICB-DCM/PEtab/blob/master/doc/documentation_data_format.md). The format specifies: 

* an SBML file with the model definition
* a measurement file with the experimental data
* a condition file that specifies different initial conditions

So this converter 1. reads the SBML file and converts it to the COPASI format, then converts the experimental data so we can use it in COPASI, and provides the mapping to the observables. Once the converter is done, and you run the parameter estimation you will get the current solution displayed. 

The benchmarks are also added as submodule to this repo. So if you do want to use those, be sure to check it out as well, running: 

	git submodule init 
	git submodule update

### Run PEtab the easy way
This project is used direclty from [basico](https://basico.rtdf.io), where we directly implement a PetabSimulator. So if you are interested in just running PEtab problems with COPASI, I recommend to 

	pip install copasi-basico[petab]
	
and then follow the [basico petab example](https://basico.readthedocs.io/en/latest/notebooks/Working_with_PEtab.html). Otherwise, this project can of course be used on its own as described below. 

### Setup
Create a new virtual environment, and then run `pip install -r requirements.txt`. This will install all the dependencies, these are: 

- numpy
- pandas
- python-copasi
- python-libsbml
- PyQt5
- pyyaml

You can also directly install the importer in one line directly from git (including the dependencies) using: 

	pip install git+https://github.com/copasi/python-petab-importer.git

and you can run directly: 

	copasi_petab_import  [<petab_dir>]  <model_name> <output_dir>

### Usage
Once installed, you can use the graphical user interface, specify the benchmark directory, select the test and the model, and you ought to be able to open the generated COPASI file directly. You do this by running: 

    python PEtab.py
    
<img src="./doc/demo.gif">
    
Alternatively you cold convert the benchmark models directly by invoking the converter: 

    python convert_petab.py <benchmark_dir> <model_name> <output_dir>
    
where:

  * `benchmark_dir` is a directory to a pe tab dir, like `./hackathon_contributions_new_data_format/Becker_Science2010`.
  * `model_name` is one of the model names in the directory like `Becker_Science2010__BaF3_Exp`. The program assumes, that the measurement data and condition data files are in the directory containing the model name (otherwise any measurement / condition file will be greedily taken)
  * `output_dir` is the directory into which the output will be written. For example '/out'. In this case at the end of the run the files `Becker_Science2010__BaF3_Exp.cps` and `Becker_Science2010__BaF3_Exp.txt` would be generated. 

Also added a bulk converter: 

    python convert_all_petab.py <base_dir> <output_dir>
    
where:

  * `base_dir` is the pe tab root dir, as in `./hackathon_contributions_new_data_format/`
  * `output_dir` the directory in which the files will be saved in

### License
Just as COPASI, the packages available on this page are provided under the 
[Artistic License 2.0](http://copasi.org/Download/License/), 
which is an [OSI](http://www.opensource.org/) approved license. This license 
allows non-commercial and commercial use free of charge.
