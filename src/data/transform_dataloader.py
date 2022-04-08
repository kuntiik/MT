from torch.utils.data import DataLoader


def transform_dl(
        dataset,
        build_batch,
        batch_tfms=None,
        build_batch_kwargs: dict = {},
        **dataloader_kwargs,
):
    """Creates collate_fn from build_batch (collate function) by decorating it with apply_batch_tfms and unload_records"""
    collate_fn = apply_batch_tfms(
        build_batch, batch_tfms=batch_tfms, **build_batch_kwargs
    )
    collate_fn = unload_records(collate_fn)
    return DataLoader(dataset=dataset, collate_fn=collate_fn, **dataloader_kwargs)


def apply_batch_tfms(build_batch, batch_tfms=None, **build_batch_kwargs):
    """This decorator function applies batch_tfms to records before passing them to build_batch"""

    def inner(records):
        if batch_tfms is not None:
            records = batch_tfms(records)
        return build_batch(records, **build_batch_kwargs)

    return inner


def unload_records(build_batch):
    """This decorator function unloads records to not carry them around after batch creation"""

    def inner(records):
        tupled_output, records = build_batch(records)
        for record in records:
            record.unload()
        return tupled_output, records

    return inner
