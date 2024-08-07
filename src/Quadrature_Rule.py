import torch
from numpy.polynomial.legendre import leggauss

class Quadrature_Rule:
    """
    Initialize the Quadrature_Rule class.
    
    Parameters:
    - collocation_points (torch.Tensor): Collocation points for integration.
    - quadrature_rule (str): The name of the quadrature rule to use.
    - number_integration_nodes (int): Number of integration nodes.
    - polynomial_degree (int): Degree of the polynomial space.
    """
    def __init__(self,
                 collocation_points: torch.Tensor,
                 quadrature_rule: str = "Gauss-Legendre", 
                 number_integration_nodes: int = 5, 
                 polynomial_degree: int = 1):
        
        self.quadrature_rule_name = quadrature_rule
        self.number_integration_nodes = number_integration_nodes
        self.polynomial_degree = polynomial_degree
        
        if self.quadrature_rule_name == "Gauss-Legendre":
            integration_nodes, integration_weights = leggauss(number_integration_nodes)
            self.integration_nodes = torch.tensor(integration_nodes, requires_grad = False)
            self.integration_weights = torch.tensor(integration_weights,  requires_grad = False)
        
        self.update_collocation_points(collocation_points)
        
    def polynomial_evaluations(self):
        """
        Evaluate polynomials at the integration nodes.
        """
        print("Computing Polynomial Evaluations...")
        with torch.no_grad():
            
            if self.polynomial_degree == 0:
                self.polynomial_evaluation = torch.ones_like(self.mapped_integration_nodes)
            
            if self.polynomial_degree == 1:
                poly_eval_positive = (self.mapped_integration_nodes - self.collocation_points[:-1].unsqueeze(1)) / self.elements_diameter.unsqueeze(1)
                poly_eval_negative = (self.collocation_points[1:].unsqueeze(1) - self.mapped_integration_nodes) / self.elements_diameter.unsqueeze(1)
                
                self.polynomial_evaluation = torch.stack([poly_eval_positive, poly_eval_negative], dim=0)
                
                value = self.polynomial_evaluation.clone()
                self.polynomial_evaluation[value > 1.0] = 0
                self.polynomial_evaluation[value < 0.0] = 0
                
                del poly_eval_positive, poly_eval_negative, value
            
    def update_collocation_points(self, 
                                  collocation_points: torch.Tensor):
        """
        Update collocation points and related attributes, such as mapped nodes 
        and weights for integration.
        
        Parameters:
        - collocation_points (torch.Tensor): The new collocation points.
        """
        print("Updating Integration Points...")
        with torch.no_grad():
            self.collocation_points = collocation_points
            
            self.elements_diameter = collocation_points[1:] - collocation_points[:-1]
            self.sum_collocation_points = collocation_points[1:] + collocation_points[:-1]
            self.number_subintervals = self.elements_diameter.size(0)
            
            self.mapped_weights = 0.5 * self.elements_diameter.unsqueeze(1) * self.integration_weights
            self.mapped_integration_nodes = 0.5 * self.elements_diameter.unsqueeze(1) * self.integration_nodes + 0.5 * self.sum_collocation_points.unsqueeze(1)
            self.mapped_integration_nodes_single_dimension = self.mapped_integration_nodes.view(-1)
            
            del collocation_points
            
            self.polynomial_evaluations()
        
    def integrate(self, 
                  function_values: torch.Tensor, 
                  multiply_by_test: bool = True):
        """
        Perform integration using the quadrature rule.
        
        Parameters:
        - function_values (torch.Tensor): Function values at the integration nodes.
        - multiply_by_test (bool): Multiply function values by test functions values (default is True).
        Returns:
        - torch.Tensor: The integral values.
        """
        function_values = function_values
        
        if(multiply_by_test == True):
        
            integral_value = torch.zeros((self.number_subintervals, self.polynomial_degree + 1))
            
            for i in range(self.polynomial_degree + 1):
                nodes_value = self.polynomial_evaluation[i, :, :] * function_values.view(self.mapped_integration_nodes.size())
                integral_value[:, i] = torch.sum(self.mapped_weights * nodes_value, dim=1)
        
        else:
            
            nodes_value = function_values.view(self.mapped_integration_nodes.size())
            integral_value = torch.sum(self.mapped_weights * nodes_value, dim=1)
        
        del function_values, nodes_value
        
        return integral_value