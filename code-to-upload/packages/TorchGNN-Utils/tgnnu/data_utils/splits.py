from abc import ABC
import torch
import math

# Defining Split Manager
class SplitManager(ABC):
    def __init__(self, dataset):
        self.device = dataset.x.device
        self.dataset = dataset
        self.perm_idx = torch.randperm(self.dataset.x.shape[0]).to(self.device)
        self.perm_selected = torch.zeros_like(self.perm_idx).bool().to(self.device)
        self.perm_class = self.dataset.y[self.perm_idx]

    def alloc(self, budget, budget_allocated="overall", stratified=False, return_cumulative=False, return_mask=True):
        """ Allocates a set of indices based on the budget and the budget allocation strategy
        """
        if budget_allocated == "overall" and stratified:
            budget = math.ceil(budget / len(torch.unique(self.dataset.y)))

        if budget_allocated == "per_class" and stratified == False:
            raise ValueError("Budget allocation per class is only possible with stratified sampling")

        if stratified == False:
            selected = self.perm_idx[~self.perm_selected][:budget]
            flipping_idx = (self.perm_selected == False).nonzero(as_tuple=True)[0][:budget]
            self.perm_selected[flipping_idx] = True

        else:
            overall_selected = []
            for class_idx in torch.unique(self.perm_class):
                cls_idx = class_idx.item()
                class_pidx = self.perm_idx[(~self.perm_selected) & (self.perm_class == cls_idx)]
                class_selected = class_pidx[:min(budget, class_pidx.shape[0])]
                overall_selected.append(class_selected)
            overall_selected = torch.concat(overall_selected)
            out = torch.zeros_like(self.perm_idx).bool()
            out[overall_selected] = True
            self.perm_selected = self.perm_selected | out[self.perm_idx]
            selected = overall_selected


        if return_cumulative:
            result = self.perm_idx[self.perm_selected]
        else:
            result = selected

        if return_mask == True:
            out = torch.zeros_like(self.perm_idx).bool()
            out[result] = True
            return out
        else:
            out = result
            return out
        
    def shuffle_free_idxs(self):
        free_idxs = self.perm_idx[~self.perm_selected]
        new_perm_unselected = torch.randperm(free_idxs.shape[0])

        # updaing perm_idx
        self.perm_idx[~self.perm_selected] = free_idxs[new_perm_unselected]

        # updating perm_class
        free_classes = self.perm_class[~self.perm_selected]
        self.perm_class[~self.perm_selected] = free_classes[new_perm_unselected]
        