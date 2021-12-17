#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <math.h> 

using namespace cv;
using namespace std;

// Hyperparameters to configure before running. All values are configured for VLP16 LIDAR.
#define OBJS 8 // Number of planes to detect
#define ITER 500 // Ransac number of iterations
#define SSIZE 4 // How many elements to pick out: sample size
#define THRESHOLD 0.3 // RANSAC error distance threshold in meter
#define NEIGHDIST 1 // Neighborhood size to pick from during estimation
#define TOLERANCE 600 // Tolerance for neighborhood density fom middle point
#define INTENSITYTHRESH 15.5 // Threshold for outputting objects above intensity level

vector<float> splitString(string arg, char splitter) // Split a string on given character and return as an array
{
	stringstream test(arg);
	string segment;
	vector<float> seglist;

	while (std::getline(test, segment, splitter))
	{
		if (segment != "")
		{
			seglist.push_back(std::stof(segment));
		}
	}
	return seglist;
}

vector<pair<Point3f, int>> readFile(char** argv, bool header) // Read points into a vector
{
	vector<pair<Point3f, int>> pairs;
	fstream newfile;
	newfile.open(argv[1], ios::in);

	if (newfile.is_open())
	{
		string tp;
		while (getline(newfile, tp))
		{
			if (header)
			{
				header = false;
			}
			else
			{
				vector<float> data = splitString(tp, ','); // Split by character
				if (data.size() == 10)
				{
					pair<Point3f, int> elem;
					elem.first = Point3f((float)data[7], (float)data[8], (float)data[9]);
					elem.second = (int)data[0];
					
					float distFromOrigo = sqrt(elem.first.x * elem.first.x + elem.first.y * elem.first.y + elem.first.z * elem.first.z);
					
					if (distFromOrigo > 3.5)
					{
						pairs.push_back(elem);
					}
				}
			}
		}
		newfile.close();
	}
	else
	{
		cout << "Cannot open file.txt" << endl;
		return pairs;
	}
	return pairs;
}

vector<pair<Point3f, int>> removeElements(vector<pair<Point3f,int>> set, vector<bool> mask, string mode)
{
	vector<pair<Point3f, int>> result;
	for (int i = 0; i < mask.size(); i++) {

		if (mask.at(i)) // True --> inlier
		{
			if (mode == "inlier")
			{
				result.push_back(set.at(i));
			}
		}
		else // False --> outlier
		{
			if (mode == "outlier") // Only add if we want to remove outliers
			{
				result.push_back(set.at(i));
			}
		}
	}
	return result;
}

vector<bool> getEmptyMask(int size) // Creates an empty inlier/outlier mask full of false elements
{
	vector<bool> result;
	for (int i = 0; i < size; i++)
	{
		result.push_back(false);
	}
	return result;
}

vector<Point3f> getFirst(vector<pair<Point3f, int>> points) // Retrieve points from the pairs
{
	vector<Point3f> result;
	for (int i = 0; i < points.size(); i++)
	{
		result.push_back(points.at(i).first);
	}
	return result;
}

vector<int> getSecond(vector<pair<Point3f, int>> points)
{
	vector<int> result;
	for (int i = 0; i < points.size(); i++)
	{
		result.push_back(points.at(i).second);
	}
	return result;
}

Point3f centerOfGravity(vector<Point3f> pts) // Calculate the center of gravity given a set of points in 3D
{
	Point3f cog;
	int num = pts.size();

	for (int i = 0; i < num; i++)
	{
		cog.x += pts.at(i).x;
		cog.y += pts.at(i).y;
		cog.z += pts.at(i).z;
	}

	cog.x /= num;
	cog.y /= num;
	cog.z /= num;

	return cog;
}

float* estimatePlane(vector<Point3f> pts) // Fit a plane to a subset of points
{
	int num = pts.size();
	Point3f cog = centerOfGravity(pts);
	Mat Q(num, 4, CV_32F);

	for (int idx = 0; idx < num; idx++) // Form a linear set of equations
	{
		Point3d pt = pts.at(idx);
		Q.at<float>(idx, 0) = pt.x - cog.x;
		Q.at<float>(idx, 1) = pt.y - cog.y;
		Q.at<float>(idx, 2) = pt.z - cog.z;
		Q.at<float>(idx, 3) = 1;
	}

	Mat mtx = Q.t() * Q;

	Mat evals, evecs;
	eigen(mtx, evals, evecs);

	float A = evecs.at<float>(3, 0);
	float B = evecs.at<float>(3, 1);
	float C = evecs.at<float>(3, 2);
	float D = (-A * cog.x - B * cog.y - C * cog.z);

	float norm = sqrt(A * A + B * B + C * C);

	float* ret = new float[4];

	ret[0] = A / norm;
	ret[1] = B / norm;
	ret[2] = C / norm;
	ret[3] = D / norm;

	return ret;
}

vector<pair<Point3f, int>> getPointsInRange(vector<pair<Point3f, int>> subset, Point3f midPoint, float t)
{
	vector<pair<Point3f, int>> result;
	int num = subset.size();
	for (int i = 0; i < num; i++)
	{
		Point3f refPoint = subset.at(i).first;
		float distance = sqrt(pow(midPoint.x - refPoint.x, 2) + pow(midPoint.y - refPoint.y, 2) + pow(midPoint.z - refPoint.z, 2));
		if (distance < t)
		{
			result.push_back(subset.at(i));
		}
	}
	return result;
}

void updateTolTries(int &tol, int &tries, int tolThresh) // This will guarantee that the algorithm finishes every time
{
	if (tries >= tolThresh)
	{
		tol -= 50;
		tries = 0;
	}
	else
	{
		tries++;
	}
}

void printLoadingBar(int i)
{
	if (i % (int)(ITER / 20) == 0)
	{
		cout << ">";
	}
}

vector<Point3f> pickRandomElements(int n, vector<pair<Point3f, int>> points) // Pick random elements from a set of pair points
{
	int num = points.size() - 1;
	vector<Point3f> resultPick;
	vector<int> result;

	while (result.size() < n)
	{
		int randi = (rand() % (num + 1)); // Generate a random number 

		if (result.size() == 0)
		{
			result.push_back(randi);
		}
		else
		{
			bool flag = false;
			for (int i = 0; i < result.size(); i++)
			{
				if (result.at(i) == randi)
				{
					flag = true;
				}
			}
			if (!flag)
			{
				result.push_back(randi);
			}
		}
	}

	for (int i = 0; i < n; i++)
	{

		int ind = result.at(i);
		Point3f newPt = points.at(ind).first;
		resultPick.push_back(newPt);
	}

	return resultPick;
}

vector<float> calculateDistances(vector<pair<Point3f, int>> points, float* params) // Calculates plane-line distance
{
	vector<float> result;
	for (int i = 0; i < points.size(); i++)
	{
		Point3f pt = points.at(i).first;
		float A = params[0];
		float B = params[1];
		float C = params[2];
		float D = params[3];
		result.push_back(fabs(A * pt.x + B * pt.y + C * pt.z + D));
	}
	return result;
}

int flagInliers(vector<bool> &mask, vector<float> distances) // Flag true on mask wherever an inlier is found according to distance vector
{
	int inlierCount = 0;
	for (int i = 0; i < distances.size(); i++) 
	{
		if (distances.at(i) < THRESHOLD) // Inlier
		{
			mask.at(i) = true;
		}
		if (mask.at(i)) // This is separate -> on further iterations there will be trues which were classified as inliers on a previous iteration
		{
			inlierCount++;
		}
	}
	return inlierCount;
}

void outputCloud(vector<pair<Point3f, int>> points, int i, string message) // Output the points into a pointCloud
{
	// Get name of file 
	char filename[10];
	std::string s = std::to_string(i);
	char const *pchar = s.c_str();
	strcpy_s(filename, "o");
	strcat_s(filename, "u");
	strcat_s(filename, "t");
	strcat_s(filename, pchar);
	strcat_s(filename, ".");
	strcat_s(filename, "x");
	strcat_s(filename, "y");
	strcat_s(filename, "z");

	// Output file into a file Outx.xyz x=(0,1,..,num)
	int num = points.size();
	ofstream myfile;
	myfile.open(filename);
	myfile << message << endl;

	for (int i = 0; i < num; i++) // Iterate over the points and append each coordinate to the file
	{
		Point3f currpt = points.at(i).first;
		myfile << currpt.x << " " << currpt.y << " " << currpt.z << " " << endl;
	}

	myfile << endl;
	myfile.close();
}

float estimateAverageIntensity(vector<pair<Point3f, int>> points) // Estimate the average intensity values for an object
{
	float result = 0;
	int num = points.size();
	for (int i = 0; i < num; i++)
	{
		result += points.at(i).second;
	}
	result /= num;
	return result;
}

float estimateMedianIntensity(vector<pair<Point3f, int>> points)
{
	int n = points.size();
	vector<int> arr = getSecond(points);
	sort(arr.begin(), arr.end());
	
	if (n % 2 == 0)
	{
		return (arr[n / 2 - 1] + arr[n / 2]) / 2;
	}
	return arr[n / 2];
}

void processRefPoints(float intensity, vector<pair<Point3f, int>> refPoints, vector<pair<Point3f, int>> &finalset)
{
	if (intensity > INTENSITYTHRESH)
	{
		for (int i = 0; i < refPoints.size(); i++)
		{
			finalset.push_back(refPoints.at(i));
		}
	}
}

int main(int argc, char** argv)
{
	cout << "Starting robust estimation..." << endl;

	// Setup initial values
	vector<pair<Point3f, int>> points = readFile(argv, true);
	vector<pair<Point3f, int>> subset = points;
	vector<pair<Point3f, int>> finalset;

	int num = points.size();
	vector<bool> fullMask = getEmptyMask(num);

	// Number of planes to detect
	for (int h = 0; h < OBJS; h++) // Iterate over objects
	{	
		cout << "Subset size: " << subset.size() << endl;

		int tries = 0;
		int tol = TOLERANCE;
		int bestInlierNum = 0;
		float* bestParams = new float[4];
		vector<bool> bestInlierSubMask = getEmptyMask(subset.size());

		for (int i = 0; i < ITER; i++) // RANSAC iteration
		{	
			int rangeCount = 0;
			vector<pair<Point3f, int>> rangeset;
			
			do 
			{
				Point3f midPoint = pickRandomElements(2, subset).at(1); // Pick a random point

				rangeset = getPointsInRange(subset, midPoint, NEIGHDIST); // Filter out points in a NEIGHDIST meter distance from it
				
				rangeCount = rangeset.size(); // This will guarantee to have enough samples in the range
				
				updateTolTries(tol, tries, 1000);
			} 
			while (rangeCount < tol); // Make sure that we pick a point from a dense neighborhood 

			vector<Point3f> ranSample = pickRandomElements(SSIZE, rangeset); // Randomly pick 4 elements from the filtered subset

			float* params = estimatePlane(ranSample); // Fit a plane to the 4 randomly picked points

			vector<float> distances = calculateDistances(subset, params); // Calculate distances for inlier thresholding

			vector<bool> subMask = getEmptyMask(subset.size()); // Create an inlier -> true / outlier -> false mask --> all false now

			int inlierCount = flagInliers(subMask, distances); // Set inliers from false to true

			if (inlierCount > bestInlierNum) // Assign new best variables if necessary
			{
				bestInlierNum = inlierCount;
				bestInlierSubMask = subMask;
				bestParams = params;
			}

			printLoadingBar(i); // This part will print the loading bar
		}
		cout << endl;

		vector<pair<Point3f, int>> inlierSet = removeElements(subset, bestInlierSubMask, "inlier"); // Get inlier points for current plane params

		vector<Point3f> inlierPoints = getFirst(inlierSet); // Get point set for inlier points

		float* inlierParams = estimatePlane(inlierPoints); // Reestimate the plane on the inlier set

		vector<float> distancesFull = calculateDistances(points, inlierParams); // Calculate distances for reestimated plane

		vector<bool> refMask = getEmptyMask(num); // Create an empty mask for newly selected points

		int inlierCountFull = flagInliers(fullMask, distancesFull); // Flag inliers for all iterations

		int inlierCountRef = flagInliers(refMask, distancesFull); // Flag inliers for current iteration 

		subset = removeElements(points, fullMask, "outlier"); // Keep only outliers for next round of estimation

		vector<pair<Point3f, int>>  refPoints = removeElements(points, refMask, "inlier"); // Keep only inliers

		float avgIntensity = estimateAverageIntensity(refPoints); // Estimate the average intensity of inlier points

		float medIntensity = estimateMedianIntensity(refPoints); // Estimate the median intensity of inlier points

		processRefPoints(medIntensity, refPoints, finalset); // Add the component to the final result based on inlier intensity

		cout << "Plane params: " << inlierParams[0] << ", " << inlierParams[1] << ", " << inlierParams[2] << ", " << inlierParams[3] << endl;
		cout << "Total inliers: " << inlierCountFull << endl;
		cout << "Current inliers: " << inlierCountRef << endl; 
		cout << "Average intensity: " << avgIntensity << endl;
		cout << "Median intensity: " << medIntensity << endl;
		cout << "Finished processing object " << h+1 << endl << endl;

		outputCloud(subset, h+1, "# Outlier set in an iteration after removing component."); // Output the full outlier pointcloud 
		outputCloud(refPoints, 11 + h, "# Reference points belonging to a specific component."); // Output the reference point cloud
	}

	outputCloud(finalset, 99, "Final assembled points"); // Output the final point cloud 

	return 0;
}