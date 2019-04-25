/*
[将Mulan上的(xml,arff)数据集转换为Matlab版ML-ODM可以读取的txt数据集]

*/

#include "data_processing.h"

void transferDataOnMulan(string xmlFilePath,string arffFilePath,bool isSparse,string featureFilePath,string labelFilePath) {
	ifstream arffFile;
	ofstream featureFile, labelFile;
	string fileLine;
	int featureNum, labelNum;
	int attributeNum = 0;
	vector<string> attributeVals;
	vector<string> sparseAttributeVals;
	vector<string> attributeValItems;
	int nonZeroIndex;
	vector<int> nonZeroIndexs;
	string newFeatureLine;
	string newLabelLine;
	bool hasLabel;
	int i,j;

	arffFile.open(arffFilePath);
	if (!arffFile.is_open()) {
		cout << "Arff file open failed" << endl;
		return;
	}
	featureFile.open(featureFilePath);
	if (!featureFile.is_open()) {
		cout << "Feature file create failed" << endl;
		return;
	}
	labelFile.open(labelFilePath);
	if (!labelFile.is_open()) {
		cout << "Label file create failed" << endl;
		return;
	}

	// count labels
	labelNum = countLabelsFromXml(xmlFilePath);

	// count attributes
	while (getline(arffFile, fileLine)) {
		if (fileLine.compare("@data") == 0) break;
		else {
			if (!fileLine.empty() && fileLine.substr(1, 9).compare("attribute") == 0) attributeNum++;
		}
	}

	// calculate features
	featureNum = attributeNum - labelNum;

	// read the data line and transfer
	while (getline(arffFile, fileLine)) {
		hasLabel = false;
		if (!isSparse) {
			// split the data line
			attributeVals.clear();
			SplitString(fileLine, attributeVals, ",");

			// build new feature line
			newFeatureLine = "";
			for (i = 0; i < featureNum; i++) {
				newFeatureLine += attributeVals[i] + " ";
			}
			newFeatureLine.pop_back();

			// build new label line
			newLabelLine = "";
			for (i = 0; i < labelNum; i++) {
				newLabelLine += attributeVals[i + featureNum] + " ";
				if (hasLabel == false) {
					if (attributeVals[i + featureNum].compare("0") != 0) hasLabel = true;
				}
			}
			newLabelLine.pop_back();
		}
		else {
			// split the data line
			sparseAttributeVals.clear();
			fileLine = fileLine.substr(1, fileLine.size() - 2);    //delete"{" and "}"
			SplitString(fileLine, sparseAttributeVals, ",");

			// get nonzero attribute value indexs
			nonZeroIndexs.clear();
			for (i = 0; i < sparseAttributeVals.size(); i++) {
				attributeValItems.clear();
				SplitString(sparseAttributeVals[i], attributeValItems, " ");
				nonZeroIndex = stoi(attributeValItems[0]);
				nonZeroIndexs.push_back(nonZeroIndex);
			}

			// build new feature line
			newFeatureLine = "";
			for (i = 0, j = 0; i < featureNum; i++) {
				if (j < nonZeroIndexs.size() && i == nonZeroIndexs[j]) {
					newFeatureLine += "1 ";
					j++;
				}
				else {
					newFeatureLine += "0 ";
				}
			}
			newFeatureLine.pop_back();

			// bulid new label line
			newLabelLine = "";
			for (i = 0; i < labelNum; i++) {
				if (j < nonZeroIndexs.size() && i + featureNum == nonZeroIndexs[j]) {
					newLabelLine += "1 ";
					j++;
					hasLabel = true;
				}
				else {
					newLabelLine += "0 ";
				}
			}
			newLabelLine.pop_back();
		}
		if (hasLabel) {
			featureFile << newFeatureLine << endl;
			labelFile << newLabelLine << endl;
		}
	}
	arffFile.close();
	featureFile.close();
	labelFile.close();
}

int main() {
	/*
	string datasetNames = "emotions";
	string state = "test";
	bool isSparse = false;
	string xmlFilePath = "./MulanDatasets/" + datasetNames + ".xml";
	string trainArffFilePath = "./MulanDatasets/" + datasetNames + "_" + state + ".arff";
	string featureFilePath = "./MulanDatasets/" + datasetNames + "_" + state + "_feature.txt";
	string labelFilePath = "./MulanDatasets/" + datasetNames + "_" + state + "_label.txt";

	transferDataOnMulan(xmlFilePath, trainArffFilePath, isSparse, featureFilePath, labelFilePath);
	*/
	vector<string> datasetNames(8);
	datasetNames[0] = "emotions", datasetNames[1] = "yeast", datasetNames[2] = "birds", datasetNames[3] = "scene";
	datasetNames[4] = "genbase", datasetNames[5] = "medical", datasetNames[6] = "Corel5k", datasetNames[7] = "enron";
	vector<string> states(2);
	states[0] = "train", states[1] = "test";
	vector<bool> isSparse(8, false);
	isSparse[5] = true, isSparse[7] = true;
	
	for (int i = 0; i < datasetNames.size(); i++) {
		for (int j = 0; j < states.size(); j++) {
			string xmlFilePath = "./MulanDatasets/" + datasetNames[i] + ".xml";
			string trainArffFilePath = "./MulanDatasets/" + datasetNames[i] + "_" + states[j] + ".arff";
			string featureFilePath = "./MulanDatasets/" + datasetNames[i] + "_" + states[j] + "_feature.txt";
			string labelFilePath = "./MulanDatasets/" + datasetNames[i] + "_" + states[j] + "_label.txt";

			transferDataOnMulan(xmlFilePath, trainArffFilePath, isSparse[i], featureFilePath, labelFilePath);
		}
		cout << datasetNames[i] << " finished!" << endl;
	}

	system("pause");
}
