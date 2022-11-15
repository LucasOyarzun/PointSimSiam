import torch
from datasets import ShapeNet55Dataset
import os
from models import get_model
from optimizers import get_optimizer, LR_Scheduler

from arguments import parse_args

import polyscope as ps


def main(args):
    if args.vis_debug:
        ps.init()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = ShapeNet55Dataset(config=args,
                                        npoints=1024,
                                        split='train')

    if args.vis_debug:
        idx = 20
        pc1 = ps.register_point_cloud("Transform 1", train_dataset[idx][2])
        pc2 = ps.register_point_cloud("Transform 2", train_dataset[idx][3])
        pc3 = ps.register_point_cloud("Original", train_dataset[idx][4])
        ps.show()
    
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.train.batch_size)

    model = get_model(args.model)
    model = torch.nn.DataParallel(model)

    optimizer = get_optimizer(args.train.optimizer.name, model, 
                                lr=args.train.base_lr*args.train.batch_size/256,
                                momentum=args.train.optimizer.momentum,
                                weight_decay=args.train.optimizer.weight_decay)
    
    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256,
        args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256, 
        len(train_loader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )

    
    accuracy = 0
    for epoch in range(args.train.stop_at_epoch):
        model.train()

        for index, (tax, obj_id, data1, data2, data3) in enumerate(train_loader):
            model.zero_grad()
            data1 = data1.to(device, non_blocking=True)
            data2 = data2.to(device, non_blocking=True)
            
            data1 = data1.transpose(2,1).contiguous()
            data2 = data2.transpose(2,1).contiguous()
            
            data_dict = model(data1, data2)
            loss = data_dict['loss'].mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            print(f'{index}/{len(train_loader)} - Loss: {loss.item()}')
        
        model_path = os.path.join(args.ckpt_dir, f'{args.name}_{epoch+1}')
        torch.save({'epoch':epoch+1,'state_dict':model.module.state_dict()}, model_path)

        print(f'Model saved to {model_path}')
    
    #if args.eval is not False:
    #    args.eval_from = model_path
    #    linear_eval(args) 

if __name__=='__main__':
    args = parse_args()
    main(args)


# EXperimentos

#Podrimaos usar el mismo protocolo que point mae para los experimentos, quermos solucionar le mismo probblema.
# Asi como esta la red uno ya podria pasarle shapenet y hacer que se entrene y guardar ese último modelo despues de x epochs.
# Luego tomaremos otro dataset (Ej modelnet) y lo que haremos es hacer fine tuning. Tomamos la red preentrenada inicialmente,
# Ahora si le colocamos una cabeza de clasficacion, y entrenamos toda la red para clasifciar modelNEt, se mediran los resultados con modelNet

# Sobre utilizarlo con otro tipo de problemas

# Point MAE hace fine tuning sobre modelnet, sobre scanobject y luego fine tuning para segmentacion de datos 3d(Identificar partes de un objeto)
# En documento final podemos proponer trabajar simplemente con clasificacion de modelnet y scanobject.

# En la propuesta podemos hablar sobre el trabajo que hacemos comprender simsiam, como inyectarle un modelo de nube de puntos y primer experimento de hacer una corrida de entrenamiento de la red
