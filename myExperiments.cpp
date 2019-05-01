//头文件
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <time.h>
#include <math.h>

#include "mlc_svm.h"
#include "mlc_evaluation.h"

//宏定义
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

//函数声明
void parse_mlc_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_mlc_problem(const char *filename);
void do_mlc_svm_cross_validation(const char *filename);

//全局变量
struct svm_node *x;
int max_nr_attr = 64;

struct mlc_svm_model* model;

static char *line = NULL;
static int max_line_len;

char control_file[256] = { "../Control/Control_MLCSVM_Train.txt" };
char predict_control_file[256] = { "../Control/Control_MLCSVM_Predict.txt" };
int  argc;
char **argv;

struct train_parameter  param;		// set by parse_command_line
struct mlc_problem mlcprob;
struct svm_node *x_space;
int    *y_space;
int    cross_validation;
int    nr_fold;


//功能函数
void print_null(const char *s) {}
//---------------------------------------------------------------------
void read_mlc_predict_control_file()
{

	char line[1024];
	int  nlin = 0, maxlen = 0;;
	int  i;


	FILE *fp = fopen(predict_control_file, "r");

	printf("Control File : %s\n", predict_control_file);

	if (fp == NULL) {
		printf("Can not find your control file .......\n");
		exit(1);
	}

	printf("**********************************************\n");
	printf(">Contents in the control file\n");

	do {
		if (fgets(line, 1024, fp) == NULL)break;
		if (line[0] == '#')break;
		printf("> %s", line);
		if (maxlen < (int)strlen(line))maxlen = (int)strlen(line);
		nlin++;
	} while (1);
	rewind(fp);

	printf("**********************************************\n");

	if (nlin < 3) {
		printf("There are less than three lines in your control file !\n");
		exit(1);
	}

	if (nlin > 3) {
		printf("There are too many lines in your control file !\n");
		exit(1);
	}


	argc = nlin + 1;


	argv = (char **)malloc(argc * sizeof(char *));

	maxlen += 2;
	for (i = 0; i < argc; i++) {
		argv[i] = (char *)malloc(maxlen * sizeof(char));
	}

	strcpy(argv[0], "SVMML1.00");
	for (i = 1; i < argc; i++) {
		fscanf(fp, "%s", argv[i]);
	}

	fclose(fp);

	return;
}


//-------------------------------------------------------------------------------------------------
static char* readline(FILE *input)
{
	int len;

	if (fgets(line, max_line_len, input) == NULL)
		return NULL;

	while (strrchr(line, '\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *)realloc(line, max_line_len);
		len = (int)strlen(line);
		if (fgets(line + len, max_line_len - len, input) == NULL)
			break;
	}
	return line;
}

//--------------------------------------------------------------------------------------------------
void exit_input_error(int line_num)
{
	fprintf(stderr, "Wrong input format at line %d\n", line_num);
	exit(1);
}

//--------------------------------------------------------------------------------------------------
void mlc_svm_predict_file(FILE *input, FILE *output)
{
	int i, k, total = 0;
	int true_size = 0, pred_size = 0;
	int svm_type = model->param.svm_type;
	int nr_class = model->nr_class;
	int *true_labels = (int *)malloc(nr_class * sizeof(int));
	int *pred_labels = (int *)malloc(nr_class * sizeof(int));
	double *ypred = NULL;
	struct ranking_indexes  rind;
	struct instance_indexes iind;
	struct label_indexes    mind;
	struct mlc_indexes ind;
	long   tstart, tfinish;
	double threshold = 0.0;
	int j, nr_instance = 0;
	double *yp = NULL, *lt = NULL, *lp = NULL, *yorg = NULL;


	max_line_len = 1024;
	line = (char *)malloc(max_line_len * sizeof(char));

	ind.rank_ind.coverage = 0.0;
	ind.rank_ind.one_error = 0.0;
	ind.rank_ind.precision1 = 0.0;
	ind.rank_ind.rankingloss = 0.0;
	ind.rank_ind.iserror = 0.0;
	ind.rank_ind.errorsetsize = 0.0;
	ind.rank_ind.auc = 0.0;

	ind.inst_ind.hammingloss = 0.0;
	ind.inst_ind.accuracy = 0.0;
	ind.inst_ind.Fmeasure = 0.0;
	ind.inst_ind.precision2 = 0.0;
	ind.inst_ind.recall = 0.0;
	ind.inst_ind.subset_accuracy = 0.0;

	tstart = get_runtime_ms();

	nr_instance = 0;
	while (readline(input) != NULL) nr_instance++;
	printf("Total test instances = %d\n", nr_instance);
	rewind(input);

	printf("Begin to predict............................\n");

	yp = (double *)malloc(nr_instance*nr_class * sizeof(double));
	lt = (double *)malloc(nr_instance*nr_class * sizeof(double));
	lp = (double *)malloc(nr_instance*nr_class * sizeof(double));


	fprintf(output, "Name: "
		"{[%d discriminant fucntion values]|threshold} {[Predicted lables]==>>[True labels]}\n", nr_class);

	total = 0;
	while (readline(input) != NULL) //read a line from the test file
	{

		char *idx, *val, *label, *endptr, *name, *lal;
		int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		i = 0;
		k = 0;

		name = strtok(line, " \t"); // example title
		//printf("\n%s ",name);
		fprintf(output, "%s: ", name);
		label = strtok(NULL, " \t"); // label string

		//get the features of each example
		while (1)
		{
			if (i >= max_nr_attr - 1)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct svm_node *) realloc(x, max_nr_attr * sizeof(struct svm_node));
			}

			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if (val == NULL)
				break;
			errno = 0;
			x[i].index = (int)strtol(idx, &endptr, 10);
			if (endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total + 1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val, &endptr);
			if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total + 1);

			++i;
		}
		x[i].index = -1;

		//get the labels of each example
		//printf("Labels ====> %s\n",label);
		if (strchr(label, ',')) {//  labels>=2
			k = 0;
			lal = strtok(label, ",");
			do {
				//printf("Lal 1 => %s\n",lal);
				if (lal == NULL)break;

				errno = 0;
				true_labels[k] = strtol(lal, &endptr, 10);
				//printf("Endptr => %s\n",endptr);
				if (errno != 0 || endptr == lal)	exit_input_error(i + 1);
				k++;
				lal = strtok(NULL, ",");
			} while (1);
		}
		else { // only one label
			k = 0;
			true_labels[k] = strtol(label, &endptr, 10);
			if (endptr == label)exit_input_error(i + 1);
			k++;
		}

		true_size = k;

		//printf("Proba= %d\n",predict_probability);
		ypred = mlc_svm_predict_one(model, x);

		//fprintf(output,"\n");
		//for(j=0;j<nr_class;j++)fprintf(output,"%lf ",ypred[j]);fprintf(output,"\n");

		threshold = 0.0;
		if (model->regress == 1) {
			threshold = model->thr0;
			for (j = 0; j < nr_class; j++)threshold += model->wthr[j] * ypred[j];
		}

		pred_size = 0;
		fprintf(output, "{[");
		for (j = 0; j < nr_class; j++) {
			if (j < nr_class - 1)fprintf(output, "%lf;", ypred[j]);
			else fprintf(output, "%lf", ypred[j]);

			if (ypred[j] >= threshold) {
				pred_labels[pred_size] = j + 1;
				pred_size++;
			}

		}
		fprintf(output, "]|%lf} {[", threshold);


		if (pred_size == 0) {// at least we would take one label for each example
			if (model->param.one == 1) {
				k = 0;
				for (j = 1; j < nr_class; j++) {
					if (ypred[j] > ypred[k])k = j;
				}
				pred_labels[0] = k + 1;
				pred_size = 1;
			}
		}

		if (pred_size == 0)fprintf(output, " ");
		else if (pred_size == 1)fprintf(output, "%d", pred_labels[0]);
		else {
			for (j = 0; j < pred_size - 1; j++)fprintf(output, "%d,", pred_labels[j]);
			j = pred_size - 1;
			fprintf(output, "%d", pred_labels[j]);
		}

		fprintf(output, "]==>>[");

		if (true_size == 0)fprintf(output, " ");
		else if (true_size == 1)fprintf(output, "%d", true_labels[0]);
		else {
			for (j = 0; j < true_size - 1; j++)fprintf(output, "%d,", true_labels[j]);
			j = true_size - 1;
			fprintf(output, "%d", true_labels[j]);
		}

		fprintf(output, "]}\n");

		rind = evaluate_ranking_indexes(nr_class, true_size, true_labels, ypred);
		iind = evaluate_instance_indexes(nr_class, true_size, true_labels, pred_size, pred_labels);

		ind.rank_ind.coverage += rind.coverage;
		ind.rank_ind.one_error += rind.one_error;
		ind.rank_ind.precision1 += rind.precision1;
		ind.rank_ind.rankingloss += rind.rankingloss;
		ind.rank_ind.iserror += rind.iserror;
		ind.rank_ind.errorsetsize += rind.errorsetsize;
		ind.rank_ind.auc += rind.auc;


		ind.inst_ind.hammingloss += iind.hammingloss;
		ind.inst_ind.accuracy += iind.accuracy;
		ind.inst_ind.Fmeasure += iind.Fmeasure;
		ind.inst_ind.precision2 += iind.precision2;
		ind.inst_ind.recall += iind.recall;
		ind.inst_ind.subset_accuracy += iind.subset_accuracy;

		for (j = 0; j < nr_class; j++) {
			lt[j*nr_instance + total] = -1;
			lp[j*nr_instance + total] = -1;
			yp[j*nr_instance + total] = ypred[j];
		}
		for (j = 0; j < true_size; j++)lt[nr_instance*(true_labels[j] - 1) + total] = 1;
		for (j = 0; j < pred_size; j++)lp[nr_instance*(pred_labels[j] - 1) + total] = 1;

		++total;
		if (total % 100 == 0)printf("-->%d", total);


	}
	printf("-->%d\n", total);

	mind = evaluate_label_indexes(nr_instance, nr_class, lt, yp, lp);

	ind.rank_ind.coverage /= (double)(total);
	ind.rank_ind.one_error /= (double)(total);
	ind.rank_ind.precision1 /= (double)(total);
	ind.rank_ind.rankingloss /= (double)(total);
	ind.rank_ind.iserror /= (double)(total);
	ind.rank_ind.errorsetsize /= (double)(total);
	ind.rank_ind.auc /= (double)(total);

	ind.inst_ind.hammingloss /= (double)(total);
	ind.inst_ind.accuracy /= (double)(total);
	ind.inst_ind.Fmeasure /= (double)(total);
	ind.inst_ind.precision2 /= (double)(total);
	ind.inst_ind.recall /= (double)(total);
	ind.inst_ind.subset_accuracy /= (double)(total);

	tfinish = get_runtime_ms();
	ind.traintime = 0.0;
	ind.testtime = tfinish - tstart;

	printf("************************************************************************\n");
	printf("Results:\n");

	printf("Testing time (ms)  =  %.0lf\n", ind.testtime);

	printf("Seven measures based on ranking ...........................\n");
	printf("Coverage           =  %.5lf\n", ind.rank_ind.coverage);
	printf("One error          =  %.5lf\n", ind.rank_ind.one_error);
	printf("Average precision  =  %.5lf\n", ind.rank_ind.precision1);
	printf("Ranking loss       =  %.5lf\n", ind.rank_ind.rankingloss);
	printf("Is error           =  %.5lf\n", ind.rank_ind.iserror);
	printf("Error set size     =  %.5lf\n", ind.rank_ind.errorsetsize);
	printf("AUC                =  %.5lf\n", ind.rank_ind.auc);

	printf("Six measures based on instances .......................\n");
	printf("Hamming loss     =  %.5lf\n", ind.inst_ind.hammingloss);
	printf("Accuracy         =  %.5lf\n", ind.inst_ind.accuracy);
	printf("F Measure        =  %.5lf\n", ind.inst_ind.Fmeasure);
	printf("Precision        =  %.5lf\n", ind.inst_ind.precision2);
	printf("Recall           =  %.5lf\n", ind.inst_ind.recall);
	printf("Subset accuracy  =  %.5lf\n", ind.inst_ind.subset_accuracy);

	printf("Eight measures based on labels...........................\n");
	printf("Macro precision  = %.5lf\n", mind.macroprecision);
	printf("Macro recall     = %.5lf\n", mind.macrorecall);
	printf("Macro F1         = %.5lf\n", mind.macroF1);
	printf("Macro AUC        = %.5lf\n", mind.macroAUC);

	printf("Micro precision  = %.5lf\n", mind.microprecision);
	printf("Micro recall     = %.5lf\n", mind.microrecall);
	printf("Micro F1         = %.5lf\n", mind.microF1);
	printf("Micro AUC        = %.5lf\n", mind.microAUC);

	printf("************************************************************************\n");

	free(true_labels);
	free(pred_labels);
	free(ypred);
	free(lp);
	free(lt);
	free(yp);
}

//**********************************************************************************************
void read_mlc_control_file()
{

	char line[1024];
	int  nlin0 = 0, ml = 0;
	int  i, p = 0;


	FILE *fp = fopen(control_file, "r");

	printf("Control File : %s\n", control_file);
	printf("*************************************************\n");
	printf(">Contents in the control file:\n");

	if (fp == NULL) {
		printf("Can not find your control file .......\n");
		exit(1);
	}

	do {
		if (fgets(line, 1024, fp) == NULL)break;
		if (line[0] == '#')break;
		if (line[0] == '-')p++;
		if (ml < (int)strlen(line))ml = strlen(line);
		printf(">  %s", line);
		nlin0++;
	} while (1);
	rewind(fp);

	printf("********************************************\n");

	if ((nlin0 - p) < 2) {
		printf("Warning: please add a training or model data file in the control file\n");
		exit(1);
	}
	if ((nlin0 - p) > 2) {
		printf("Warning: please remove a training and model data file from the control file\n");
		exit(1);
	}

	argc = 2 * p + 3;
	argv = (char **)malloc(argc * sizeof(char *));

	for (i = 0; i < argc; i++) {
		argv[i] = (char *)malloc(ml * sizeof(char));
	}

	strcpy(argv[0], "SVMML100");
	for (i = 0; i < p; i++) {
		fscanf(fp, "%s %s\n", argv[2 * i + 1], argv[2 * i + 2]);
	}

	fscanf(fp, "%s\n", argv[argc - 2]);
	fscanf(fp, "%s\n", argv[argc - 1]);

	fclose(fp);

	return;
}

void exit_with_help()
{
	printf(
		"Usage: svmml-train\n"
		"options:\n"
		"-s svm_type : set type of multi-label SVM (default 2)\n"
		"	0 -- Rank-SVM\n"
		"   1 -- SVM-ML (i.e., Rank-SVMz)\n"
		"   2 -- Rank-CVM\n"
		"   3 -- Rank-CVMz\n"

		"-j method_type : set optimization method type (default 0)\n"
		"   0 -- FWM (Frank-Wolfe method)\n"
		"   1 -- BCDM (block coordinate descent method)\n"

		"-b block size : set the size of block for BCDM\n"

		"-t kernel_type : set type of kernel function (default 1)\n"
		"	0 -- linear: u'*v\n"
		"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
		"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
		"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
		"-d degree : set degree in kernel function (default 2)\n"
		"-g gamma : set gamma in kernel function (1.0)\n"
		"-r coef0 : set coef0 in kernel function (default 0)\n"

		"-c cost : set the parameter C of multi-label SVMs (default 1)\n"

		"-w regress : linear ridge regression model for threshold function for Rank-SVM/CVM only (default 1)\n"
		"   0 -- no model for SVM-ML mode (default)\n"
		"   1 -- minimal Hamming loss with a large threshold for each training instance\n"
		"   2 -- minimal Hamming loss with a small threshold for each training instance\n"
		"   3 -- maximal Chi2 to detect a threshold for each training instance\n"
		"   4 -- maximal accuracy to detect a threhsold for each training instance\n"
		"   5 -- maximal F1 to detect a threshold for each training instance\n"
		"-p rdidge : a regularization constant for the ridge regression (default 1.0e-6)\n"

		"-m iterations : set maximal iterations for Frank-Wolfe method (default 50)\n"
		"-n tolerance : set tolerance of termination criterion for relative norm (default 1.0e-3) in CDM\n"
		"-e tolerance : set tolerance of termination criterion for graient*direction(default 1.0e-3) in FWM\n"
		"-f tolerance : set tolerance for step size (default 1.0e-6) in FWM\n"
		"-h tolerance : set tolerance for support vectors (default 1.0e-4)\n"
		"-a threshold: set a gradient threshold for RBCDM (default 1.0)\n"

		"-o one : one predicted label at least for 1 (default 0)\n"

		"-v k: k-fold cross validation mode\n"
		"-z seed : random seed for k-fold cross validation\n"
		"-i verbose(default 0)\n"

		"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

//-------------------------------------------------------------------------------------------------------
void evaluate_cross_validation_fold(const struct mlc_problem prob, double **target, int *yfold,
	int nr_fold, long *trtm, long *tetm, double *threshold,
	struct mlc_indexes *mean, struct mlc_indexes *std)
{
	int i, j, m, kk, k, mm;
	int sr, *rl = NULL;
	int sp, *pl = NULL;
	struct ranking_indexes  rind;
	struct instance_indexes iind;
	struct mlc_indexes *ind_fold = Malloc(struct mlc_indexes, nr_fold);
	int *count = Malloc(int, nr_fold);

	//intialization
	for (i = 0; i < nr_fold; i++) {
		ind_fold[i].rank_ind.coverage = 0.0;
		ind_fold[i].rank_ind.one_error = 0.0;
		ind_fold[i].rank_ind.precision1 = 0.0;
		ind_fold[i].rank_ind.rankingloss = 0.0;
		ind_fold[i].rank_ind.iserror = 0.0;
		ind_fold[i].rank_ind.errorsetsize = 0.0;
		ind_fold[i].rank_ind.auc = 0.0;

		ind_fold[i].inst_ind.hammingloss = 0.0;
		ind_fold[i].inst_ind.accuracy = 0.0;
		ind_fold[i].inst_ind.Fmeasure = 0.0;
		ind_fold[i].inst_ind.precision2 = 0.0;
		ind_fold[i].inst_ind.recall = 0.0;
		ind_fold[i].inst_ind.subset_accuracy = 0.0;

		ind_fold[i].mami_ind.macroAUC = 0.0;
		ind_fold[i].mami_ind.macroF1 = 0.0;
		ind_fold[i].mami_ind.macroprecision = 0.0;
		ind_fold[i].mami_ind.macrorecall = 0.0;
		ind_fold[i].mami_ind.microAUC = 0.0;
		ind_fold[i].mami_ind.microF1 = 0.0;
		ind_fold[i].mami_ind.microprecision = 0.0;
		ind_fold[i].mami_ind.microrecall = 0.0;

	}

	for (i = 0; i < nr_fold; i++)count[i] = 0;

	//calculate ranking and instance based indexes for each example
	for (i = 0; i < prob.l; i++) {
		kk = yfold[i] - 1;
		count[kk]++;

		sr = 0;
		while (1) {
			m = prob.y[i][sr];
			if (m < 0)break;
			sr++;
		}
		rl = prob.y[i];
		rind = evaluate_ranking_indexes(prob.c, sr, rl, target[i]);

		sp = 0;
		for (j = 0; j < prob.c; j++)if (target[i][j] >= threshold[i])sp++;
		if (sp > 0) {
			pl = Malloc(int, sp);
			sp = 0;
			for (j = 0; j < prob.c; j++) {
				if (target[i][j] >= threshold[i]) {
					pl[sp] = j + 1;
					sp++;
				}
			}
		}

		//* take one label at least if param.one=1
		else {
			if (param.one == 1) {
				sp = 1;
				pl = Malloc(int, sp);
				k = 0;
				for (j = 1; j < prob.c; j++) {
					if (target[i][j] > target[i][k])k = j;
				}
				pl[0] = k + 1;
			}

		}


		iind = evaluate_instance_indexes(prob.c, sr, rl, sp, pl);
		if (sp > 0)free(pl);

		ind_fold[kk].rank_ind.coverage += rind.coverage;
		ind_fold[kk].rank_ind.one_error += rind.one_error;
		ind_fold[kk].rank_ind.precision1 += rind.precision1;
		ind_fold[kk].rank_ind.rankingloss += rind.rankingloss;
		ind_fold[kk].rank_ind.iserror += rind.iserror;
		ind_fold[kk].rank_ind.errorsetsize += rind.errorsetsize;
		ind_fold[kk].rank_ind.auc += rind.auc;

		ind_fold[kk].inst_ind.hammingloss += iind.hammingloss;
		ind_fold[kk].inst_ind.accuracy += iind.accuracy;
		ind_fold[kk].inst_ind.Fmeasure += iind.Fmeasure;
		ind_fold[kk].inst_ind.precision2 += iind.precision2;
		ind_fold[kk].inst_ind.recall += iind.recall;
		ind_fold[kk].inst_ind.subset_accuracy += iind.subset_accuracy;
	}

	//calculate the indexes for each fold
	for (i = 0; i < nr_fold; i++) {
		ind_fold[i].rank_ind.coverage /= (double)(count[i]);
		ind_fold[i].rank_ind.one_error /= (double)(count[i]);
		ind_fold[i].rank_ind.precision1 /= (double)(count[i]);
		ind_fold[i].rank_ind.rankingloss /= (double)(count[i]);
		ind_fold[i].rank_ind.iserror /= (double)(count[i]);
		ind_fold[i].rank_ind.errorsetsize /= (double)(count[i]);
		ind_fold[i].rank_ind.auc /= (double)(count[i]);

		ind_fold[i].inst_ind.hammingloss /= (double)(count[i]);
		ind_fold[i].inst_ind.accuracy /= (double)(count[i]);
		ind_fold[i].inst_ind.Fmeasure /= (double)(count[i]);
		ind_fold[i].inst_ind.precision2 /= (double)(count[i]);
		ind_fold[i].inst_ind.recall /= (double)(count[i]);
		ind_fold[i].inst_ind.subset_accuracy /= (double)(count[i]);

		ind_fold[i].traintime = trtm[i];
		ind_fold[i].testtime = tetm[i];
	}


	//calculate label-based indexes
	for (kk = 0; kk < nr_fold; kk++) {
		double *l_true = (double *)malloc(count[kk] * prob.c * sizeof(double));
		double *l_pred = (double *)malloc(count[kk] * prob.c * sizeof(double));
		double *y_pred = (double *)malloc(count[kk] * prob.c * sizeof(double));
		for (j = 0; j < count[kk] * prob.c; j++) { l_true[j] = -1; l_pred[j] = -1; }

		mm = 0;
		for (i = 0; i < prob.l; i++) {
			if (yfold[i] != kk + 1)continue;

			sr = 0;
			while (1) {
				m = prob.y[i][sr];
				if (m < 0)break;
				l_true[(m - 1)*count[kk] + mm] = 1.0;
				sr++;
			}

			sp = 0;
			for (j = 0; j < prob.c; j++) {
				if (target[i][j] >= threshold[i]) {
					l_pred[j*count[kk] + mm] = 1.0;
					y_pred[j*count[kk] + mm] = target[i][j];
					sp++;
				}
			}

			// one label at least for param.one=1
			if (sp == 0) {
				if (param.one == 1) {
					k = 0;
					for (j = 1; j < prob.c; j++) {
						if (target[i][j] > target[i][k])k = j;
					}
					l_pred[k*count[kk] + mm] = 1.0;
				}
			}


			for (j = 0; j < prob.c; j++)y_pred[j*count[kk] + mm] = target[i][j];

			mm++;
		}
		ind_fold[kk].mami_ind = evaluate_label_indexes(count[kk], prob.c, l_true, y_pred, l_pred);
		free(y_pred);
		free(l_true);
		free(l_pred);
	}


	//calculate the mean and std 
	mean->rank_ind.coverage = 0.0;
	mean->rank_ind.one_error = 0.0;
	mean->rank_ind.precision1 = 0.0;
	mean->rank_ind.rankingloss = 0.0;
	mean->rank_ind.iserror = 0.0;
	mean->rank_ind.errorsetsize = 0.0;
	mean->rank_ind.auc = 0.0;

	mean->inst_ind.hammingloss = 0.0;
	mean->inst_ind.accuracy = 0.0;
	mean->inst_ind.Fmeasure = 0.0;
	mean->inst_ind.precision2 = 0.0;
	mean->inst_ind.recall = 0.0;
	mean->inst_ind.subset_accuracy = 0.0;

	mean->mami_ind.macroAUC = 0.0;
	mean->mami_ind.macroprecision = 0.0;
	mean->mami_ind.macrorecall = 0.0;
	mean->mami_ind.macroF1 = 0.0;
	mean->mami_ind.microAUC = 0.0;
	mean->mami_ind.microprecision = 0.0;
	mean->mami_ind.microrecall = 0.0;
	mean->mami_ind.microF1 = 0.0;

	mean->traintime = 0.0;
	mean->testtime = 0.0;

	std->rank_ind.coverage = 0.0;
	std->rank_ind.one_error = 0.0;
	std->rank_ind.precision1 = 0.0;
	std->rank_ind.rankingloss = 0.0;
	std->rank_ind.iserror = 0.0;
	std->rank_ind.errorsetsize = 0.0;
	std->rank_ind.auc = 0.0;

	std->inst_ind.hammingloss = 0.0;
	std->inst_ind.accuracy = 0.0;
	std->inst_ind.Fmeasure = 0.0;
	std->inst_ind.precision2 = 0.0;
	std->inst_ind.recall = 0.0;
	std->inst_ind.subset_accuracy = 0.0;

	std->mami_ind.macroAUC = 0.0;
	std->mami_ind.macroprecision = 0.0;
	std->mami_ind.macrorecall = 0.0;
	std->mami_ind.macroF1 = 0.0;
	std->mami_ind.microAUC = 0.0;
	std->mami_ind.microprecision = 0.0;
	std->mami_ind.microrecall = 0.0;
	std->mami_ind.microF1 = 0.0;


	std->traintime = 0.0;
	std->testtime = 0.0;

	for (i = 0; i < nr_fold; i++) {
		mean->rank_ind.coverage += ind_fold[i].rank_ind.coverage;
		mean->rank_ind.one_error += ind_fold[i].rank_ind.one_error;
		mean->rank_ind.precision1 += ind_fold[i].rank_ind.precision1;
		mean->rank_ind.rankingloss += ind_fold[i].rank_ind.rankingloss;
		mean->rank_ind.iserror += ind_fold[i].rank_ind.iserror;
		mean->rank_ind.errorsetsize += ind_fold[i].rank_ind.errorsetsize;
		mean->rank_ind.auc += ind_fold[i].rank_ind.auc;

		mean->inst_ind.hammingloss += ind_fold[i].inst_ind.hammingloss;
		mean->inst_ind.accuracy += ind_fold[i].inst_ind.accuracy;
		mean->inst_ind.Fmeasure += ind_fold[i].inst_ind.Fmeasure;
		mean->inst_ind.precision2 += ind_fold[i].inst_ind.precision2;
		mean->inst_ind.recall += ind_fold[i].inst_ind.recall;
		mean->inst_ind.subset_accuracy += ind_fold[i].inst_ind.subset_accuracy;

		mean->mami_ind.macroAUC += ind_fold[i].mami_ind.macroAUC;
		mean->mami_ind.macroprecision += ind_fold[i].mami_ind.macroprecision;
		mean->mami_ind.macrorecall += ind_fold[i].mami_ind.macrorecall;
		mean->mami_ind.macroF1 += ind_fold[i].mami_ind.macroF1;
		mean->mami_ind.microAUC += ind_fold[i].mami_ind.microAUC;
		mean->mami_ind.microprecision += ind_fold[i].mami_ind.microprecision;
		mean->mami_ind.microrecall += ind_fold[i].mami_ind.microrecall;
		mean->mami_ind.microF1 += ind_fold[i].mami_ind.microF1;


		mean->traintime += ind_fold[i].traintime;
		mean->testtime += ind_fold[i].testtime;
	}

	mean->rank_ind.coverage /= (double)(nr_fold);
	mean->rank_ind.one_error /= (double)(nr_fold);
	mean->rank_ind.precision1 /= (double)(nr_fold);
	mean->rank_ind.rankingloss /= (double)(nr_fold);
	mean->rank_ind.iserror /= (double)(nr_fold);
	mean->rank_ind.errorsetsize /= (double)(nr_fold);
	mean->rank_ind.auc /= (double)(nr_fold);

	mean->inst_ind.hammingloss /= (double)(nr_fold);
	mean->inst_ind.accuracy /= (double)(nr_fold);
	mean->inst_ind.Fmeasure /= (double)(nr_fold);
	mean->inst_ind.precision2 /= (double)(nr_fold);
	mean->inst_ind.recall /= (double)(nr_fold);
	mean->inst_ind.subset_accuracy /= (double)(nr_fold);

	mean->mami_ind.macroAUC /= (double)(nr_fold);
	mean->mami_ind.macroprecision /= (double)(nr_fold);
	mean->mami_ind.macrorecall /= (double)(nr_fold);
	mean->mami_ind.macroF1 /= (double)(nr_fold);
	mean->mami_ind.microAUC /= (double)(nr_fold);
	mean->mami_ind.microprecision /= (double)(nr_fold);
	mean->mami_ind.microrecall /= (double)(nr_fold);
	mean->mami_ind.microF1 /= (double)(nr_fold);

	mean->traintime /= (double)(nr_fold);
	mean->testtime /= (double)(nr_fold);

	for (i = 0; i < nr_fold; i++) {
		std->rank_ind.coverage += pow((ind_fold[i].rank_ind.coverage - mean->rank_ind.coverage), 2.0);
		std->rank_ind.one_error += pow((ind_fold[i].rank_ind.one_error - mean->rank_ind.one_error), 2.0);
		std->rank_ind.precision1 += pow((ind_fold[i].rank_ind.precision1 - mean->rank_ind.precision1), 2.0);
		std->rank_ind.rankingloss += pow((ind_fold[i].rank_ind.rankingloss - mean->rank_ind.rankingloss), 2.0);
		std->rank_ind.iserror += pow((ind_fold[i].rank_ind.iserror - mean->rank_ind.iserror), 2.0);
		std->rank_ind.errorsetsize += pow((ind_fold[i].rank_ind.errorsetsize - mean->rank_ind.errorsetsize), 2.0);
		std->rank_ind.auc += pow((ind_fold[i].rank_ind.auc - mean->rank_ind.auc), 2.0);

		std->inst_ind.hammingloss += pow((ind_fold[i].inst_ind.hammingloss - mean->inst_ind.hammingloss), 2.0);
		std->inst_ind.accuracy += pow((ind_fold[i].inst_ind.accuracy - mean->inst_ind.accuracy), 2.0);
		std->inst_ind.Fmeasure += pow((ind_fold[i].inst_ind.Fmeasure - mean->inst_ind.Fmeasure), 2.0);
		std->inst_ind.precision2 += pow((ind_fold[i].inst_ind.precision2 - mean->inst_ind.precision2), 2.0);
		std->inst_ind.recall += pow((ind_fold[i].inst_ind.recall - mean->inst_ind.recall), 2.0);
		std->inst_ind.subset_accuracy += pow((ind_fold[i].inst_ind.subset_accuracy - mean->inst_ind.subset_accuracy), 2.0);

		std->mami_ind.macroAUC += pow((ind_fold[i].mami_ind.macroAUC - mean->mami_ind.macroAUC), 2.0);
		std->mami_ind.macroprecision += pow((ind_fold[i].mami_ind.macroprecision - mean->mami_ind.macroprecision), 2.0);
		std->mami_ind.macrorecall += pow((ind_fold[i].mami_ind.macrorecall - mean->mami_ind.macrorecall), 2.0);
		std->mami_ind.macroF1 += pow((ind_fold[i].mami_ind.macroF1 - mean->mami_ind.macroF1), 2.0);
		std->mami_ind.microAUC += pow((ind_fold[i].mami_ind.microAUC - mean->mami_ind.microAUC), 2.0);
		std->mami_ind.microprecision += pow((ind_fold[i].mami_ind.microprecision - mean->mami_ind.microprecision), 2.0);
		std->mami_ind.microrecall += pow((ind_fold[i].mami_ind.microrecall - mean->mami_ind.microrecall), 2.0);
		std->mami_ind.microF1 += pow((ind_fold[i].mami_ind.microF1 - mean->mami_ind.microF1), 2.0);


		std->traintime += pow((ind_fold[i].traintime - mean->traintime), 2.0);
		std->testtime += pow((ind_fold[i].testtime - mean->testtime), 2.0);
	}

	std->rank_ind.coverage /= (double)(nr_fold);
	std->rank_ind.one_error /= (double)(nr_fold);
	std->rank_ind.precision1 /= (double)(nr_fold);
	std->rank_ind.rankingloss /= (double)(nr_fold);
	std->rank_ind.iserror /= (double)(nr_fold);
	std->rank_ind.errorsetsize /= (double)(nr_fold);
	std->rank_ind.auc /= (double)(nr_fold);

	std->inst_ind.hammingloss /= (double)(nr_fold);
	std->inst_ind.accuracy /= (double)(nr_fold);
	std->inst_ind.Fmeasure /= (double)(nr_fold);
	std->inst_ind.precision2 /= (double)(nr_fold);
	std->inst_ind.recall /= (double)(nr_fold);
	std->inst_ind.subset_accuracy /= (double)(nr_fold);

	std->mami_ind.macroAUC /= (double)(nr_fold);
	std->mami_ind.macroprecision /= (double)(nr_fold);
	std->mami_ind.macrorecall /= (double)(nr_fold);
	std->mami_ind.macroF1 /= (double)(nr_fold);
	std->mami_ind.microAUC /= (double)(nr_fold);
	std->mami_ind.microprecision /= (double)(nr_fold);
	std->mami_ind.microrecall /= (double)(nr_fold);
	std->mami_ind.microF1 /= (double)(nr_fold);

	std->traintime /= (double)(nr_fold);
	std->testtime /= (double)(nr_fold);

	std->rank_ind.coverage = sqrt(std->rank_ind.coverage);
	std->rank_ind.one_error = sqrt(std->rank_ind.one_error);
	std->rank_ind.precision1 = sqrt(std->rank_ind.precision1);
	std->rank_ind.rankingloss = sqrt(std->rank_ind.rankingloss);
	std->rank_ind.iserror = sqrt(std->rank_ind.iserror);
	std->rank_ind.errorsetsize = sqrt(std->rank_ind.errorsetsize);
	std->rank_ind.auc = sqrt(std->rank_ind.auc);

	std->inst_ind.hammingloss = sqrt(std->inst_ind.hammingloss);
	std->inst_ind.accuracy = sqrt(std->inst_ind.accuracy);
	std->inst_ind.Fmeasure = sqrt(std->inst_ind.Fmeasure);
	std->inst_ind.precision2 = sqrt(std->inst_ind.precision2);
	std->inst_ind.recall = sqrt(std->inst_ind.recall);
	std->inst_ind.subset_accuracy = sqrt(std->inst_ind.subset_accuracy);

	std->mami_ind.macroAUC = sqrt(std->mami_ind.macroAUC);
	std->mami_ind.macroprecision = sqrt(std->mami_ind.macroprecision);
	std->mami_ind.macrorecall = sqrt(std->mami_ind.macrorecall);
	std->mami_ind.macroF1 = sqrt(std->mami_ind.macroF1);
	std->mami_ind.microAUC = sqrt(std->mami_ind.microAUC);
	std->mami_ind.microprecision = sqrt(std->mami_ind.microprecision);
	std->mami_ind.microrecall = sqrt(std->mami_ind.microrecall);
	std->mami_ind.microF1 = sqrt(std->mami_ind.microF1);

	std->traintime = sqrt(std->traintime);
	std->testtime = sqrt(std->testtime);

	free(ind_fold);
	free(count);

	return;
}
//----------------------------------------------------------------------------------------
void do_mlc_svm_cross_validation(const char *file_name)
{
	int i;
	int *yfold = Malloc(int, mlcprob.l);
	double *threshold = Malloc(double, mlcprob.l);
	double **target = NULL;
	FILE *fp = fopen(file_name, "w");
	long tstart, tfinish;
	long *trtime = Malloc(long, nr_fold);
	long *tetime = Malloc(long, nr_fold);


	tstart = get_runtime_ms();

	target = mlc_svm_cross_validation(&mlcprob, &param, nr_fold, yfold, threshold, trtime, tetime);

	printf("Cross Validation OK\n");


	struct mlc_indexes mean, std;

	evaluate_cross_validation_fold(mlcprob, target, yfold, nr_fold, trtime, tetime,
		threshold, &mean, &std);


	tfinish = get_runtime_ms();

	printf("Training Time = %ld (ms)\n", tfinish - tstart);

	printf("Please check the results in the model file \n");

	fprintf(fp, "The results for k-fold cross validation\n");

	fprintf(fp, "Overall computational Time = %ld (ms)\n", tfinish - tstart);

	fprintf(fp, "Training time (ms) == %.0lf (+-) %.0lf\n", mean.traintime, std.traintime);
	fprintf(fp, "Testing time (ms)  == %.0lf (+-) %.0lf\n", mean.testtime, std.testtime);

	fprintf(fp, "\nSeven ranking-based measures\n");
	fprintf(fp, "Coverage           == %.5lf (+-) %.5lf\n", mean.rank_ind.coverage, std.rank_ind.coverage);
	fprintf(fp, "One error          == %.5lf (+-) %.5lf\n", mean.rank_ind.one_error, std.rank_ind.one_error);
	fprintf(fp, "Average Precision  == %.5lf (+-) %.5lf\n", mean.rank_ind.precision1, std.rank_ind.precision1);
	fprintf(fp, "Ranking Loss       == %.5lf (+-) %.5lf\n", mean.rank_ind.rankingloss, std.rank_ind.rankingloss);
	fprintf(fp, "Is error           == %.5lf (+-) %.5lf\n", mean.rank_ind.iserror, std.rank_ind.iserror);
	fprintf(fp, "Error set size     == %.5lf (+-) %.5lf\n", mean.rank_ind.errorsetsize, std.rank_ind.errorsetsize);
	fprintf(fp, "AUC                == %.5lf (+-) %.5lf\n", mean.rank_ind.auc, std.rank_ind.auc);

	fprintf(fp, "\nSix instance-based measures\n");
	fprintf(fp, "Hamming Loss     == %.5lf (+-) %.5lf\n", mean.inst_ind.hammingloss, std.inst_ind.hammingloss);
	fprintf(fp, "Accuracy         == %.5lf (+-) %.5lf\n", mean.inst_ind.accuracy, std.inst_ind.accuracy);
	fprintf(fp, "F1 Measure       == %.5lf (+-) %.5lf\n", mean.inst_ind.Fmeasure, std.inst_ind.Fmeasure);
	fprintf(fp, "Precision        == %.5lf (+-) %.5lf\n", mean.inst_ind.precision2, std.inst_ind.precision2);
	fprintf(fp, "Recall           == %.5lf (+-) %.5lf\n", mean.inst_ind.recall, std.inst_ind.recall);
	fprintf(fp, "Subset accuracy  == %.5lf (+-) %.5lf\n", mean.inst_ind.subset_accuracy, std.inst_ind.subset_accuracy);

	fprintf(fp, "\nSix label-based measures\n");
	fprintf(fp, "Macro precision     == %.5lf (+-) %.5lf\n", mean.mami_ind.macroprecision, std.mami_ind.macroprecision);
	fprintf(fp, "Macro recall       == %.5lf (+-) %.5lf\n", mean.mami_ind.macrorecall, std.mami_ind.macrorecall);
	fprintf(fp, "Macro F1           == %.5lf (+-) %.5lf\n", mean.mami_ind.macroF1, std.mami_ind.macroF1);
	fprintf(fp, "Macro AUC          == %.5lf (+-) %.5lf\n", mean.mami_ind.macroAUC, std.mami_ind.macroAUC);
	fprintf(fp, "Micro precison     == %.5lf (+-) %.5lf\n", mean.mami_ind.microprecision, std.mami_ind.microprecision);
	fprintf(fp, "Micro recall       == %.5lf (+-) %.5lf\n", mean.mami_ind.microrecall, std.mami_ind.microrecall);
	fprintf(fp, "Micro F1           == %.5lf (+-) %.5lf\n", mean.mami_ind.microF1, std.mami_ind.microF1);
	fprintf(fp, "Micro AUC          == %.5lf (+-) %.5lf\n", mean.mami_ind.microAUC, std.mami_ind.microAUC);

	fclose(fp);

	for (i = 0; i < mlcprob.l; i++)free(target[i]);
	free(target);
	free(yfold);
	free(trtime);
	free(tetime);
	free(threshold);

	return;

}
//-----------------------------------------------------------------------------------------------------------
void parse_mlc_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	//a b c d e f g h i j k l m n o p q r s t u v w x y z
	//- - - - - - - - - - + + - - - - - - - - + - - + + -
	int i;

	// default values
	param.svm_type = RANK_CVM; //-s
	param.method_type = FWM;   //-j

	param.block_size = 40;     //-b

	param.kernel_type = POLY; //-t
	param.degree = 1;         //-d
	param.gamma = 1.0;        //-g
	param.coef0 = 1;          //-r
	param.C = 1.0;            //-c

	param.regress = 1;       //-w
	param.ridge = 1.0e-6;    //-p

	param.maxiter = 50;       //-m
	param.normtol = 1e-3;      //-n
	param.gxdtol = 1e-3;     //-e
	param.gradhold = 1.0;    //-l
	param.lamtol = 1e-6;     //-f
	param.svtol = 1e-4;       //-h

	param.one = 0;  //-o

	cross_validation = 0; //-v
	param.seed = 1;         //-z
	param.verbose = 0;      //-i

	// parse options
	for (i = 1; i < argc; i++)
	{
		if (argv[i][0] != '-') break;
		if (++i >= argc)
			exit_with_help();
		switch (argv[i - 1][1])
		{
		case 's':
			param.svm_type = atoi(argv[i]);
			break;

		case 'j':
			param.method_type = atoi(argv[i]);
			break;

		case 'b':
			param.block_size = atoi(argv[i]);
			break;

		case 't':
			param.kernel_type = atoi(argv[i]);
			break;
		case 'd':
			param.degree = atoi(argv[i]);
			break;
		case 'g':
			param.gamma = atof(argv[i]);
			break;
		case 'r':
			param.coef0 = atof(argv[i]);
			break;

		case 'c':
			param.C = atof(argv[i]);
			break;

		case 'm':
			param.maxiter = atoi(argv[i]);
			break;
		case 'n':
			param.normtol = atof(argv[i]);
			break;
		case 'e':
			param.gxdtol = atof(argv[i]);
			break;
		case 'f':
			param.lamtol = atof(argv[i]);
			break;
		case 'a':
			param.gradhold = atof(argv[i]);
			break;

		case 'h':
			param.svtol = atof(argv[i]);
			break;

		case 'w':
			param.regress = atoi(argv[i]);
			break;
		case 'p':
			param.ridge = atof(argv[i]);
			break;

		case 'o':
			param.one = atoi(argv[i]);
			break;

		case 'q':
			mlc_svm_print_string = &print_null;
			i--;
			break;

		case 'v':
			cross_validation = 1;
			nr_fold = atoi(argv[i]);
			if (nr_fold < 2)
			{
				fprintf(stderr, "n-fold cross validation: n must >= 2\n");
				exit_with_help();
			}
			break;

		case 'z':
			param.seed = atoi(argv[i]);
			break;

		case 'i':
			param.verbose = atoi(argv[i]);
			break;

		default:
			fprintf(stderr, "Unknown option: -%c\n", argv[i - 1][1]);
			exit_with_help();
		}
	}

	// determine filenames

	if (i >= argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);
	strcpy(model_file_name, argv[i + 1]);

	//automatically detect regression model for a threshold function
	switch (param.svm_type) {
	case RANK_SVM:
	case RANK_CVM:
	case RANK_LSVM:
		if (param.regress == 0)param.regress = 1;
		break;

	case RANK_SVMZ:
	case RANK_CVMZ:
	default:
		param.regress = 0;
		break;
	}


}

// read in a problem (in similar libsvm format for multi-label classification)
void read_mlc_problem(const char *filename)
{
	int  elements, labels, max_index, inst_max_index, i, j, k;
	FILE *fp = fopen(filename, "r");
	char *endptr;
	char *idx, *val, *label, *lal, *name;

	if (fp == NULL)
	{
		fprintf(stderr, "can't open input file %s\n", filename);
		exit(1);
	}

	mlcprob.l = 0;
	elements = 0;
	labels = 0;

	max_line_len = 1024;
	line = Malloc(char, max_line_len);
	while (readline(fp) != NULL)
	{
		char *p;

		//get the example name
		p = strtok(line, " \t");
		//printf("Instance == %s\n",p);

		//get the label string
		label = strtok(NULL, " \t");
		//printf("Labels == %s\n",label);

		// count the number of features
		while (1)
		{
			p = strtok(NULL, " \t");
			//printf("Pair = %s \n",p);
			if (p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
			//printf("Element: %d\n",elements);
		}
		++elements;
		++mlcprob.l;

		//count the number of labels
		p = strtok(label, ",");
		labels++;
		while (1)
		{
			p = strtok(NULL, ",");
			if (p == NULL || *p == '\n')break;
			labels++;
		}
		labels++;
	}
	rewind(fp);

	printf("Summary for trainin data file\n");
	printf("Examples = %d\n", mlcprob.l);
	//printf("Elements = %d\n", elements);
	//printf("Labels   = %d\n", labels);
	printf("Average lables = %.4f\n", (double)(labels) / (double)(mlcprob.l) - 1.0);

	mlcprob.names = Malloc(char *, mlcprob.l);
	mlcprob.y = Malloc(int *, mlcprob.l);
	mlcprob.x = Malloc(struct svm_node *, mlcprob.l);
	x_space = Malloc(struct svm_node, elements);
	y_space = Malloc(int, labels);

	max_index = 0;
	mlcprob.c = 1;
	j = 0;
	k = 0;
	for (i = 0; i < mlcprob.l; i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		mlcprob.x[i] = &x_space[j];
		mlcprob.y[i] = &y_space[k];

		//get the name of each example
		name = strtok(line, " \t");
		mlcprob.names[i] = Malloc(char, strlen(name) + 1);
		strcpy(mlcprob.names[i], name);
		//printf("Name: %s ",mlprob.names[i]);

		//get the label string
		label = strtok(NULL, " \t");
		//printf("Label:: %s\n",label);

		//get the features of each example
		while (1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if (val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int)strtol(idx, &endptr, 10);
			if (endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i + 1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val, &endptr);
			if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i + 1);

			++j;
		}

		if (inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;

		//get the labels of each example
		//printf("Labels ====> %s\n",label);
		if (strchr(label, ',')) {// some labels
			lal = strtok(label, ",");
			do
			{
				//printf("Lal 1 => %s\n",lal);
				if (lal == NULL)break;

				errno = 0;
				y_space[k] = strtol(lal, &endptr, 10);
				//printf("Endptr => %s\n",endptr);
				if (errno != 0 || endptr == lal)
					exit_input_error(i + 1);
				if (y_space[k] > mlcprob.c)mlcprob.c = y_space[k];
				k++;

				lal = strtok(NULL, ",");


			} while (1);
		}
		else { // only one label
			y_space[k] = strtol(label, &endptr, 10);

			if (endptr == label)
				exit_input_error(i + 1);

			if (y_space[k] > mlcprob.c)mlcprob.c = y_space[k];
			k++;

		}
		y_space[k++] = -1;

	}

	//for(i=0;i<mlprob.l;i++)printf("Instance [%d] = %s\n",i+1,mlprob.names[i]);

	if (param.gamma == 0 && max_index > 0)
		param.gamma = 2.0 / max_index;

	mlcprob.d = max_index;
	printf("Classes  = %d\nDimension= %d\n", mlcprob.c, mlcprob.d);
	printf("********************************************\n");

	fclose(fp);

}

//模块函数
void predict()
{
	FILE *input, *output;
	int i;

	read_mlc_predict_control_file();
	// parse options

	i = 1;
	input = fopen(argv[i], "r");
	if (input == NULL)
	{
		fprintf(stderr, "can't open input file %s\n", argv[i]);
		exit(1);
	}

	output = fopen(argv[i + 2], "w");
	if (output == NULL)
	{
		fprintf(stderr, "can't open output file %s\n", argv[i + 2]);
		exit(1);
	}

	printf("Load model file ............................\n");
	if ((model = mlc_svm_load_model(argv[i + 1])) == 0)
	{
		fprintf(stderr, "can't open model file %s\n", argv[i + 1]);
		exit(1);
	}

	x = (struct svm_node *) malloc(max_nr_attr * sizeof(struct svm_node));

	//printf("Begin to predict............................\n");
	mlc_svm_predict_file(input, output);
	mlc_svm_destroy_model(model);
	free(x);
	free(line);
	fclose(input);
	fclose(output);
}

void train()
{
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;
	long tstart, tfinish;

	read_mlc_control_file();
	parse_mlc_command_line(argc, argv, input_file_name, model_file_name);
	read_mlc_problem(input_file_name);

	error_msg = mlc_svm_check_parameter(&mlcprob, &param);

	if (error_msg)
	{
		fprintf(stderr, "Error: %s\n", error_msg);
		exit(1);
	}

	if (cross_validation)
	{
		tstart = get_runtime_ms();

		printf("Cross validation .......................................\n");
		do_mlc_svm_cross_validation(model_file_name);
	}
	else
	{
		printf("Train...........................\n");
		tstart = get_runtime_ms();
		model = mlc_svm_train(&mlcprob, &param);

		printf("Write a model ...............................\n");

		tfinish = get_runtime_ms();

		printf("Training Time = %ld (ms)\n", tfinish - tstart);

		mlc_svm_save_model(model_file_name, model);

		printf("To verify the training set ................\n");
		mlc_svm_predict_batch(model, &mlcprob);

		printf("Destroy the model ............................\n");
		mlc_svm_destroy_model(model);
	}

	free(mlcprob.y);
	free(mlcprob.x);
	free(mlcprob.names);
	free(x_space);
	free(y_space);
	free(line);

}

int main() {
	train();
	predict();
}
