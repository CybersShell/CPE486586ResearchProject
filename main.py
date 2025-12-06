import re
from typing import Dict, List, Tuple, Optional

class MLOutputToLatex:
    """Convert machine learning model output text to LaTeX tables matching IEEE format."""
    
    def __init__(self, text: str):
        self.text = text
        self.latex_output = []
        
    def extract_model_params(self) -> Dict[str, str]:
        """Extract model parameters from the text."""
        params = {}
        
        # Find RandomForestClassifier parameters
        pattern = r'RandomForestClassifier\((.*?)\)'
        match = re.search(pattern, self.text, re.DOTALL)
        
        if match:
            params_str = match.group(1)
            # Parse individual parameters
            param_pairs = re.findall(r'(\w+)=([^,\)]+)', params_str)
            for key, value in param_pairs:
                params[key.strip()] = value.strip().strip("'\"")
        
        return params
    
    def extract_accuracy(self) -> float:
        """Extract accuracy score."""
        match = re.search(r'Accuracy:\s*([\d.]+)', self.text)
        return float(match.group(1)) if match else 0.0
    
    def extract_classification_report(self) -> List[Dict]:
        """Extract classification report data."""
        lines = self.text.split('\n')
        report_data = []
        
        in_report = False
        for line in lines:
            if 'precision' in line and 'recall' in line:
                in_report = True
                continue
            
            if in_report and line.strip():
                # Parse data lines (0, 1, accuracy, macro avg, weighted avg)
                parts = line.split()
                if len(parts) >= 4:
                    # Check if first part is a class label or average type
                    if parts[0] in ['0', '1']:
                        report_data.append({
                            'class': parts[0],
                            'precision': parts[1],
                            'recall': parts[2],
                            'f1': parts[3],
                            'support': parts[4] if len(parts) > 4 else ''
                        })
                    elif parts[0] == 'accuracy':
                        report_data.append({
                            'class': 'accuracy',
                            'precision': '',
                            'recall': '',
                            'f1': parts[1],
                            'support': parts[2] if len(parts) > 2 else ''
                        })
                    elif len(parts) >= 5 and parts[0] in ['macro', 'weighted']:
                        report_data.append({
                            'class': f"{parts[0]} {parts[1]}",
                            'precision': parts[2],
                            'recall': parts[3],
                            'f1': parts[4],
                            'support': parts[5] if len(parts) > 5 else ''
                        })
        
        return report_data
    
    def extract_class_distribution(self) -> Tuple[Dict, Dict]:
        """Extract training and test set class distributions."""
        train_dist = {}
        test_dist = {}
        
        # Extract training distribution - try multiple patterns
        # Pattern 1: 0 first, then 1
        train_match = re.search(r'Training set class distribution:.*?0\s+(\d+).*?1\s+(\d+)', 
                                self.text, re.DOTALL)
        if train_match:
            train_dist = {'0': train_match.group(1), '1': train_match.group(2)}
        else:
            # Pattern 2: 1 first, then 0
            train_match = re.search(r'Training set class distribution:.*?1\s+(\d+).*?0\s+(\d+)', 
                                    self.text, re.DOTALL)
            if train_match:
                train_dist = {'1': train_match.group(1), '0': train_match.group(2)}
        
        # Extract test distribution - try multiple patterns
        # Pattern 1: 1 first, then 0
        test_match = re.search(r'Test set class distribution:.*?1\s+(\d+).*?0\s+(\d+)', 
                               self.text, re.DOTALL)
        if test_match:
            test_dist = {'1': test_match.group(1), '0': test_match.group(2)}
        else:
            # Pattern 2: 0 first, then 1
            test_match = re.search(r'Test set class distribution:.*?0\s+(\d+).*?1\s+(\d+)', 
                                   self.text, re.DOTALL)
            if test_match:
                test_dist = {'0': test_match.group(1), '1': test_match.group(2)}
        
        # Debug: print what was found
        if not train_dist or not test_dist:
            # Try a more flexible pattern that captures everything
            train_section = re.search(r'Training set class distribution:(.*?)(?=Test set|$)', 
                                     self.text, re.DOTALL)
            test_section = re.search(r'Test set class distribution:(.*?)(?=\n\n|$)', 
                                    self.text, re.DOTALL)
            
            if train_section:
                # Extract all number pairs
                numbers = re.findall(r'[01]\s+(\d+)', train_section.group(1))
                labels = re.findall(r'([01])\s+\d+', train_section.group(1))
                if len(numbers) >= 2 and len(labels) >= 2:
                    train_dist = {labels[0]: numbers[0], labels[1]: numbers[1]}
            
            if test_section:
                # Extract all number pairs
                numbers = re.findall(r'[01]\s+(\d+)', test_section.group(1))
                labels = re.findall(r'([01])\s+\d+', test_section.group(1))
                if len(numbers) >= 2 and len(labels) >= 2:
                    test_dist = {labels[0]: numbers[0], labels[1]: numbers[1]}
        
        return train_dist, test_dist
    
    def generate_params_table(self, params: Dict[str, str], 
                             param_order: Optional[List[str]] = None) -> str:
        """Generate LaTeX table for model parameters.
        
        Args:
            params: Dictionary of parameters
            param_order: Optional list specifying the order of parameters
        """
        # Default order if not specified
        if param_order is None:
            param_order = ['class_weight', 'min_samples_leaf', 'min_samples_split', 
                          'n_estimators', 'random_state']
        
        latex = r"""\begin{table}[h!]
	\centering
	\caption{Random Forest Hyperparameters}
	\begin{tabular}{ll}
		\hline
		\textbf{Parameter} & \textbf{Value} \\
		\hline
"""
        
        # Add parameters in specified order
        for key in param_order:
            if key in params:
                latex += f"\t\t{key.replace('_', r'\_')} & {params[key]} \\\\\n"
        
        # Add any remaining parameters not in the order list
        for key, value in params.items():
            if key not in param_order:
                latex += f"\t\t{key.replace('_', r'\_')} & {value} \\\\\n"
        
        latex += r"""		\hline
	\end{tabular}
\end{table}
"""
        return latex
    
    def generate_performance_table(self, report_data: List[Dict], accuracy: float) -> str:
        """Generate LaTeX table for classification performance."""
        latex = r"""\begin{table}[h!]
	\centering
	\caption{Random Forest Classification Performance}
	\begin{tabular}{lcccc}
		\hline
		\textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Support} \\
		\hline
"""
        
        for row in report_data:
            if row['class'] == '0':
                latex += f"\t\tNormal (0) & {row['precision']} & {row['recall']} & {row['f1']} & {row['support']} \\\\\n"
            elif row['class'] == '1':
                latex += f"\t\tAttack (1) & {row['precision']} & {row['recall']} & {row['f1']} & {row['support']} \\\\\n"
        
        latex += "\t\t\\hline\n"
        latex += f"\t\t\\textbf{{Accuracy}} & \\multicolumn{{4}}{{c}}{{{accuracy:.4f}}} \\\\\n"
        
        for row in report_data:
            if 'macro' in row['class'].lower():
                latex += f"\t\tMacro Avg & {row['precision']} & {row['recall']} & {row['f1']} & {row['support']} \\\\\n"
            elif 'weighted' in row['class'].lower():
                latex += f"\t\tWeighted Avg & {row['precision']} & {row['recall']} & {row['f1']} & {row['support']} \\\\\n"
        
        latex += r"""		\hline
	\end{tabular}
\end{table}
"""
        return latex
    
    def generate_distribution_table(self, train_dist: Dict, test_dist: Dict) -> str:
        """Generate LaTeX table for class distribution."""
        latex = r"""\begin{table}[h!]
	\centering
	\caption{Training and Test Set Class Distribution}
	\begin{tabular}{lcc}
		\hline
		\textbf{Class} & \textbf{Train Count} & \textbf{Test Count} \\
		\hline
"""
        
        for class_label in ['0', '1']:
            train_count = train_dist.get(class_label, '0')
            test_count = test_dist.get(class_label, '0')
            class_name = "Normal" if class_label == '0' else "Attack"
            latex += f"\t\t{class_name} ({class_label}) & {train_count} & {test_count} \\\\\n"
        
        latex += r"""		\hline
	\end{tabular}
\end{table}
"""
        return latex
    
    def generate_confusion_matrix_figure(self, n_estimators: int = 600, 
                                        min_samples_leaf: int = 5,
                                        random_state: int = 52,
                                        combined: bool = False,
                                        figure_num: int = 1,
                                        float_position: str = 'h!') -> str:
        """Generate LaTeX figure reference for confusion matrix.
        
        Args:
            n_estimators: Number of estimators used
            min_samples_leaf: Minimum samples per leaf
            random_state: Random state used
            combined: Whether datasets were combined
            figure_num: Figure number for label
            float_position: LaTeX float position (default: 'h!' for here, strongly)
                          Options: 'h!' (here strongly), 'H' (HERE absolutely - requires float package),
                                  'htbp' (here, top, bottom, page), '!h' (override LaTeX float rules)
        """
        combined_str = "True" if combined else "False"
        latex = f"""
\\begin{{figure}}[{float_position}]
	\\centering
	\\includegraphics[width=0.45\\textwidth]{{rf_confusion_matrix-{n_estimators}-{min_samples_leaf}-{random_state}-{combined_str}.png}}
	\\caption{{Confusion matrix for the Random Forest model ({n_estimators} estimators, min\\_samples\\_leaf = {min_samples_leaf}, random\\_state = {random_state}, datasets {"" if combined else "not "}combined).}}
	\\label{{fig:rf_confusion_matrix_{figure_num}}}
\\end{{figure}}
"""
        return latex
    
    def convert(self, param_order: Optional[List[str]] = None,
                include_figure: bool = True,
                figure_params: Optional[Dict] = None,
                datasetsCombined: bool = False) -> str:
        """Convert the entire text to LaTeX.
        
        Args:
            param_order: Optional list specifying the order of parameters
            include_figure: Whether to include confusion matrix figure reference
            figure_params: Optional dict with figure parameters (n_estimators, min_samples_leaf, etc.)
        """
        # Extract data
        params = self.extract_model_params()
        accuracy = self.extract_accuracy()
        report_data = self.extract_classification_report()
        train_dist, test_dist = self.extract_class_distribution()
        
        # Generate LaTeX
        latex_output = []
        
        if params:
            latex_output.append(self.generate_params_table(params, param_order))
        
        if report_data and accuracy:
            latex_output.append(self.generate_performance_table(report_data, accuracy))
        
        if train_dist and test_dist:
            latex_output.append(self.generate_distribution_table(train_dist, test_dist))
        
        if include_figure:
            fig_params = figure_params or {}
            # Try to extract from params if not provided
            if 'n_estimators' not in fig_params and 'n_estimators' in params:
                fig_params['n_estimators'] = int(params['n_estimators'])
            if 'min_samples_leaf' not in fig_params and 'min_samples_leaf' in params:
                fig_params['min_samples_leaf'] = int(params['min_samples_leaf'])
            if 'random_state' not in fig_params and 'random_state' in params:
                fig_params['random_state'] = int(params['random_state'])
            
            latex_output.append(self.generate_confusion_matrix_figure(**fig_params))
        
        return '\n'.join(latex_output)


import os
if __name__ == "__main__":
    input_text = """Evaluating Random Forest model on test set...
Used Random Forest Classifier with the following parameters:
RandomForestClassifier(class_weight='balanced', min_samples_leaf=5,
                       min_samples_split=10, n_estimators=600, random_state=52)

Accuracy: 0.8742015613910575

Classification Report:
              precision    recall  f1-score   support

           0       0.80      0.95      0.87      9711
           1       0.96      0.82      0.88     12833

    accuracy                           0.87     22544
   macro avg       0.88      0.88      0.87     22544
weighted avg       0.89      0.87      0.87     22544

Training set class distribution:
binary_label
0    67343
1    58630
Name: count, dtype: int64

Test set class distribution:
binary_label
1    12833
0     9711
Name: count, dtype: int64
"""
for x in os.listdir():
    if x.endswith(".txt"):
        # Prints only text file present in My Folder
        with open(x, "r") as file:
            converter = MLOutputToLatex(file.read())
            combined = True if 'True' in x.rstrip(".txt").split("-")[-1] else False
            param_order = ['n_estimators', 'random_state', 'class_weight', 'min_samples_leaf', 'min_samples_split']
            latex_output = converter.convert(param_order=param_order, include_figure=True, figure_params={
                'combined': combined})
            open(f"{x.rstrip(".txt")}.tex", "w").write(latex_output)