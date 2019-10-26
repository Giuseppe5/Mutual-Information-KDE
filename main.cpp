#include <iostream>
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <fstream>
#include <vector>
#include <iomanip>
#include <math.h>


void load_csv(std::string filename, Eigen::ArrayXXd &output);

void writeToCSVfile(std::string name,Eigen::MatrixXd matrix)
{
  std::ofstream file(name.c_str());

  for(int  i = 0; i < matrix.rows(); i++){
      for(int j = 0; j < matrix.cols(); j++){
         std::string str = std::to_string(matrix(i,j));
         if(j+1 == matrix.cols()){
             file<<str;
         }else{
             file<<str<<',';
         }
      }
      file<<'\n';
  }
}

int main(int argc, char const *argv[])
{
	Eigen::ArrayXXd Input;
	Eigen::ArrayXXd Feature;

	// Choose how many samples to skip at each iteration
	int samplingTime = 3;

	// Load Data
	// Input has dimension Time X 1
	// Feature has dimension Time X N_Feature
	load_csv("Feature.csv", Feature);
	load_csv("TrainData.csv", Input);

	// Calculate effective dimension
	int dimension = ceil(Input.rows()/(double) samplingTime);

	// Create structures
	Eigen::ArrayXXd distances_feature(dimension, dimension);
	Eigen::ArrayXXd distances_input_corr(dimension,dimension);

	Eigen::ArrayXXd distances_input(dimension, dimension);
	Eigen::ArrayXXd distances_mixed(dimension, dimension);

	//Compute Kernels for 1D and 2D data
	double h_1D = 0.9 * pow((double)dimension, -1.0/5.0);
	h_1D = pow(h_1D,2);

	double h_2D =  pow((double)dimension, -1.0/6.0);
	h_2D = pow(h_2D,2);

	// Structures for variance
    double var_correction = dimension/(double)(dimension-1);
	double var_input;
	double var_feature;
	double extra_diagonal;
	Eigen::MatrixXd CovMatrix(2,2);

    var_input= var_correction * (Input.pow(2).mean()-pow(Input.mean(),2));

    // Compute distance matrix for input
	int i_aux = 0, j_aux;
	for (int i = 0; i < Input.rows(); i = i+samplingTime)
	{
		j_aux = i_aux;
		for (int j = i; j < Input.rows(); j = j+samplingTime)
		{	
			distances_input(i_aux,j_aux) = Input(i) - Input(j);
			distances_input(j_aux,i_aux) = distances_input(i_aux,j_aux);
			j_aux++;
		}
		i_aux++;
	}

	// Compute Kernel Distances
	distances_input_corr = -1.0 * distances_input.pow(2) / (2*var_input * h_1D);
	distances_input_corr = distances_input_corr.exp() / std::sqrt(2*M_PI*var_input * h_1D);
	// Compute input pdf 
	Eigen::ArrayXd C_Input = distances_input_corr.rowwise().mean();

	// Analyze Feature Matrix
	for (int z = 0; z < Feature.cols(); ++z)
	{
		// Isolate signal from one feature
		Eigen::ArrayXd signal = Feature.col(z);

		// Compute variance
		var_feature = var_correction * (signal.pow(2).mean()-pow(signal.mean(),2));
		//var_feature = var_feature;
		
		// extra_diagonal =( ( signal - signal.mean()) * (Input - Input.mean()) ).mean() / (double) (dimension-1);
		// Compute Covariance Matrix. Kernel width is 0 for extra-diagonal term
		CovMatrix(0,0) = var_input * h_2D;
		CovMatrix(0,1) = 0;
		CovMatrix(1,0) = 0;
		CovMatrix(1,1) = var_feature * h_2D;

		double determinant = std::sqrt(CovMatrix.determinant());

		// Compute inverse of diagonal matrix
		CovMatrix(0,0) = 1/(var_input * h_2D);
		CovMatrix(1,1) = 1/(var_feature * h_2D);

		Eigen::MatrixXd Support(2,1);

		int i_aux = 0, j_aux;
		for (int i = 0; i < Input.rows(); i = i+samplingTime)
		{
			j_aux = i_aux;
			for (int j = i; j < Input.rows(); j = j+samplingTime)
			{
				// Compute distance feature
				distances_feature(i_aux,j_aux) = signal(i) - signal(j);
				distances_feature(j_aux, i_aux) = distances_feature(i_aux,j_aux);

				Support(0,0) = distances_input(i_aux,j_aux);
				Support(1,0) = distances_feature(i_aux, j_aux);
				double result = (- ( (Support.transpose() * CovMatrix) * Support))(0,0);
				distances_mixed(i_aux,j_aux) = exp( result/(2) );
				distances_mixed(j_aux, i_aux) = distances_mixed(i_aux, j_aux);

				j_aux++;
			}
		i_aux++;
		}
	// Compute Kernel Feature Matrix and Kernel Joint Distance Matrix
	distances_mixed = distances_mixed / (2 * M_PI * determinant);
	distances_feature = -1.0 * distances_feature.pow(2) / (2*var_feature* h_1D);
	distances_feature = distances_feature.exp()/std::sqrt(2*M_PI * var_feature * h_1D);
	
	// Compute Feature pdf
	Eigen::ArrayXd C_Feature = distances_feature.rowwise().mean();

	// Compute Mutual Information
	Eigen::ArrayXd num =  distances_mixed.rowwise().mean();
	Eigen::ArrayXd den = C_Feature * C_Input;
	num = num.cwiseQuotient(den);
	num = num.log() / log(2);
	std::cout << "Feature: " << z << std::endl;
	std::cout <<"MI: " << num.mean() << std::endl;

	// Single Entropies
	auto feature_pdf = distances_feature.rowwise().mean().log() / log(2);
	auto input_pdf = distances_input_corr.rowwise().mean().log()/log(2);
	auto joint_pdf = distances_mixed.rowwise().mean().log()/log(2);


	std::cout << "Entropy Feature H(Y): " << -feature_pdf.mean() << std::endl;
	std::cout << "Entropy Input H(X): " << -input_pdf.mean() << std::endl;
	std::cout << "Entropy Joint H(X|Y): " << -joint_pdf.mean() << std::endl;




	}
	return 0;
}

void load_csv(std::string filename, Eigen::ArrayXXd &output)
{
    std::vector<double> vec;
    std::string buffer;
    char *tokens;
    int i, j;
    int cols = 0, line = 0;

    std::ifstream input_stream(filename.c_str());

    if (input_stream.is_open())
    {
        // // Read header
        // if (!input_stream.eof())
        // {
        //     getline(input_stream, buffer, '\n');
        //     cols = std::count(buffer.begin(), buffer.end(), ',') + 1;
        // }

        // Read data
        while (!input_stream.eof())
        {

            getline(input_stream, buffer, '\n');
            if(line == 0)
	        	cols = std::count(buffer.begin(), buffer.end(), ',') + 1;

            tokens = strtok(strdup(buffer.c_str()), ",");
            for (i = 0; (i < cols) && (tokens != NULL); i++)
            {
                vec.push_back(atof(tokens));
                tokens = strtok(NULL, ",");
            }
            line++;
        }
        // Close file
        input_stream.close();

        // Place data in matrix
        if (line > 1)
        {
            output.conservativeResize(line - 1, cols);
            for (i = 0; i < line - 1; i++)
            {
                for (j = 0; j < cols; j++)
                {
                    output(i, j) = vec[i * cols + j];
                }
            }
        }
    }
    else
    {
        std::cerr << "File " << filename << " not found" << std::endl;
        throw;
    }
}