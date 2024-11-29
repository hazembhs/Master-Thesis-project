import torch
import torch.nn.functional as F



def listwise_softmax_cross_entropy_loss(y_true, y_pred):
    """
    Listwise softmax cross entropy loss
    """
    loss = 0.0
    batch_size, num_docs = y_true.shape

    for i in range(batch_size):
        softmax_scores = F.softmax(y_pred[i], dim=0)
        log_softmax_scores = torch.log(softmax_scores)
        loss += -torch.sum(y_true[i] * log_softmax_scores)

    return loss / batch_size




def pairwise_logistic_loss(y_true, y_pred):
    batch_size = y_true.size(0)  # Get the batch size
    total_loss = 0.0

    for i in range(1, batch_size):  # Iterate over batch samples (assuming batch size > 1)
      
        loss = torch.log(1 + torch.exp(y_pred[i] - y_pred[0]))  # Compute the loss between y_pred[i] and y_pred[0]
        total_loss += loss

    return total_loss 




def pointwise_cross_entropy(y_true, y_pred):
    
    y_pred_sigmoid = torch.sigmoid(y_pred)
   
    positive_loss = -torch.sum(torch.log(y_pred_sigmoid[y_true == 1] + 1e-10))  
    negative_loss = -torch.sum(torch.log(1 - y_pred_sigmoid[y_true == 0] + 1e-10))  
    total_loss =  positive_loss + negative_loss
    return total_loss