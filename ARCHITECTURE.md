# SH Wave ML Surrogate - Architecture Diagram

## System Architecture

```mermaid
graph TB
    subgraph Data["📊 Data Layer"]
        RawData["Raw Data<br/>analytical_results.csv"]
        ProcessedData["Processed Data"]
    end

    subgraph Physics["🔬 Physics Module"]
        DispersionSolver["Dispersion Solver"]
        MaterialConstants["Material Constants"]
        ParameterSampler["Parameter Sampler"]
    end

    subgraph ML["🤖 ML Module"]
        Models["Models"]
        TrainSurrogate["Train Surrogate"]
        EvaluateModel["Evaluate Model"]
        InverseDesign["Inverse Design"]
    end

    subgraph Notebooks["📓 Notebooks"]
        DataGen["01 Data Generation"]
        Training["02 Training"]
        Visualization["03 Visualization"]
        Evaluation["04 Evaluation"]
    end

    subgraph Results["📈 Results"]
        Metrics["Metrics"]
        Plots["Plots"]
    end

    subgraph Testing["✓ Testing"]
        UnitTests["Unit Tests"]
    end

    RawData -->|Process| ProcessedData
    ProcessedData -->|Used By| DataGen
    
    DataGen -->|Generates Data| TrainSurrogate
    ParameterSampler -->|Samples Parameters| DataGen
    DispersionSolver -->|Computes Physics| DataGen
    MaterialConstants -->|Provides Constants| DispersionSolver
    
    TrainSurrogate -->|Trains| Models
    Models -->|Evaluate| EvaluateModel
    EvaluateModel -->|Produces| Metrics
    EvaluateModel -->|Creates| Plots
    
    Models -->|Uses| InverseDesign
    InverseDesign -->|Produces| Results
    
    Training -->|Uses| TrainSurrogate
    Visualization -->|Visualizes| Plots
    Evaluation -->|Uses| EvaluateModel
    
    DispersionSolver -->|Tested By| UnitTests
```

## Component Overview

### Data Layer
- **Raw Data**: Input from `analytical_results.csv`
- **Processed Data**: Cleaned and formatted data for training

### Physics Module
- **Dispersion Solver**: Computes wave dispersion characteristics
- **Material Constants**: Provides material properties
- **Parameter Sampler**: Generates parameter samples for data generation

### ML Module
- **Models**: Neural network surrogate models
- **Train Surrogate**: Training pipeline for surrogate models
- **Evaluate Model**: Model evaluation and metrics calculation
- **Inverse Design**: Inverse problem solving using trained models

### Notebooks
Sequential workflow notebooks for the complete pipeline:
1. Data Generation
2. Model Training
3. Results Visualization
4. Comprehensive Evaluation

### Results
- **Metrics**: Performance metrics and evaluation results
- **Plots**: Visualization outputs

### Testing
- **Unit Tests**: Validates physics computations
