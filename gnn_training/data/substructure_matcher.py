"""Categorical/fuzzy substructure matching."""
from typing import List, Optional, Dict, Set
import re


class SubstructureMatcher:
    """Categorical/fuzzy substructure matching for circuit families."""
    
    def __init__(self):
        """Define fuzzy categories for common substructures."""
        self.categories = {
            "differential_pair": [
                r"differential pair",
                r"diff.*pair",
                r"input.*pair",
            ],
            "active_load": [
                r"active load",
                r"current mirror",
                r"active.*load",
                r"mirror.*load",
            ],
            "load_resistors": [
                r"load resistor",
                r"interconnection resistor",
                r"resistor.*load",
                r"passive load",
            ],
            "bias_network": [
                r"bias.*network",
                r"bias.*current",
                r"current source",
                r"bias.*cell",
            ],
            "source_degeneration": [
                r"source.*degen",
                r"tail.*resistor",
                r"emitter.*resistor",
            ],
            "output_stage": [
                r"output.*stage",
                r"output buffer",
            ],
        }
    
    def extract_category(self, substructure_name: str) -> Optional[str]:
        """
        Fuzzy match substructure name to a category.
        
        Args:
            substructure_name: e.g., "M0-M1 differential pair"
        
        Returns:
            Category name if matched, None otherwise
        """
        name_lower = str(substructure_name).lower()
        
        for category, patterns in self.categories.items():
            for pattern in patterns:
                if re.search(pattern, name_lower, re.IGNORECASE):
                    return category
        
        return None
    
    def get_categorical_substructures(self, substructure_list: List[str]) -> Set[str]:
        """
        Convert a list of substructure names to their categorical representatives.
        
        Args:
            substructure_list: list of substructure names
        
        Returns:
            Set of unique categories found
        """
        categories = set()
        for substruct in substructure_list:
            cat = self.extract_category(substruct)
            if cat:
                categories.add(cat)
        return categories
    
    def get_overlap_count(self, substructs1: List[str], substructs2: List[str]) -> int:
        """
        Count overlapping categories between two circuits.
        
        Args:
            substructs1: list of substructures in circuit 1
            substructs2: list of substructures in circuit 2
        
        Returns:
            Number of overlapping categories
        """
        cats1 = self.get_categorical_substructures(substructs1)
        cats2 = self.get_categorical_substructures(substructs2)
        return len(cats1 & cats2)
    
    def get_overlap_ratio(self, substructs1: List[str], substructs2: List[str]) -> float:
        """
        Get overlap ratio (Jaccard similarity).
        
        Returns:
            overlap_count / (total_unique_categories) in range [0, 1]
        """
        cats1 = self.get_categorical_substructures(substructs1)
        cats2 = self.get_categorical_substructures(substructs2)
        
        if not cats1 and not cats2:
            return 1.0
        if not cats1 or not cats2:
            return 0.0
        
        intersection = len(cats1 & cats2)
        union = len(cats1 | cats2)
        return intersection / union if union > 0 else 0.0
