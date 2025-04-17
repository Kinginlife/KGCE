import re

import networkx as nx
from lxml import etree
from lxml.etree import _Element
from networkx import DiGraph, path_graph

from kgce import SubTask, evaluator
from kgce.actions.android_actions import execute_adb


# def get_xml_etree(env) -> _Element | None:
#     print(111111111)
#     xml_str = execute_adb("shell uiautomator dump /dev/tty", env)
#
#     if "UI hierchary dumped to: /dev/tty" not in xml_str:
#
#         return None
#     xml_str = xml_str.removesuffix("UI hierchary dumped to: /dev/tty")
#     return etree.fromstring(xml_str.encode("utf-8"))

def get_xml_etree(env) -> _Element | None:
    # 在设备上保存 XML 文件
    execute_adb("shell uiautomator dump /sdcard/window_dump.xml", env)

    # 将文件拉取到主机
    execute_adb("pull /sdcard/window_dump.xml .", env)

    # 读取 XML 文件
    try:
        with open("window_dump.xml", "r", encoding="utf-8") as f:
            xml_str = f.read()
    except FileNotFoundError:
        print("未找到 window_dump.xml 文件")
        return None

    # 返回解析后的 XML 树
    return etree.fromstring(xml_str.encode("utf-8"))


@evaluator(env_name="android", local=True)
def check_contain_input_text(text: str, env) -> bool:
    print("check_contain_input_text", text)
    if env.trajectory:
        action_name, params, _ = env.trajectory[-1]
        if action_name == "write_text" and text.lower() in params["text"].lower():
            print("return true")
            return True
    return False




@evaluator(env_name="android", local=True)
def check_contain_input_text_multiple(text: str, env) -> bool:
    if env.trajectory:
        for action_name, params, _ in env.trajectory:
            if action_name == "write_text" and text in params["text"].lower():
                return True
    return False


@evaluator(env_name="android")
def check_contain_contact(name: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    title_node = root.xpath(
        '//node[@resource-id="com.android.contacts:id/photo_touch_intercept_overlay"]'
    )
    if not title_node:
        return False
    if title_node[0].get("content-desc") != name:
        return False
    info_node = root.xpath('//*[@class="android.widget.RelativeLayout"]')
    if not info_node:
        return False
    print("info node checked")
    mail_node = None
    for node in info_node:
        desc = node.get("content-desc")
        if "Email" in desc:
            mail_node = node
    if mail_node is None:
        return False
    real_mail_node = mail_node.xpath(
        '//*[@resource-id="com.android.contacts:id/header"]'
    )
    if not real_mail_node:
        return False
    context = real_mail_node[0].get("text")
    print("context get")
    pattern = re.compile(r"^\w+@\w+.com")
    if pattern.match(context):
        return True
    return False


#这个指令出现在android_subtasks.py文件中，把这个函数修改成这样
@evaluator(env_name="android")
def check_current_package_name(name: str, env) -> bool:
    print("check_current_package_name:",name)
    result = execute_adb(
        r'shell dumpsys activity | findstr "top-activity"', env
    )
    return name in result


@evaluator(env_name="android", local=True)
def check_ocr_results(text: str, env) -> bool:
    return text in env.ocr_results


@evaluator(env_name="android")
def check_current_message_page(title: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    title_node = root.xpath(
        '//node[@resource-id="com.google.android.apps.messaging:id/conversation_title"]'
    )
    if title_node:
        return title == title_node[0].get("text")
    else:
        return False


@evaluator(env_name="android")
def check_message_text_box_contain(text: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    text_box_node = root.xpath(
        '//node[@resource-id="com.google.android.apps.messaging:id/compose_message_text"]'
    )
    if text_box_node:
        return text.lower() in text_box_node[0].get("text").lower()
    else:
        return False


@evaluator(env_name="android")
def check_message_text_box_empty(env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    text_box_node = root.xpath(
        '//node[@resource-id="com.google.android.apps.messaging:id/compose_message_text"]'
    )
    if not text_box_node:
        return False
    if text_box_node[0].get("text").strip() == "Text message":
        return True
    else:
        return False


@evaluator(env_name="android")
def check_send_message(title: str, message: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    title_node = root.xpath(
        '//node[@resource-id="com.google.android.apps.messaging:id/conversation_title"]'
    )
    if not title_node or title != title_node[0].get("text"):
        return False
    messages_node = root.xpath(
        '//node[@resource-id="com.google.android.apps.messaging:id/message_text"]'
    )
    for node in messages_node:
        if message in node.get("text"):
            return True
    return False


@evaluator(env_name="android")
def check_note_content(content: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    title_node = root.xpath(
        '//node[@resource-id="com.google.android.keep:id/editable_title"]'
    )
    if not title_node:
        return False
    if title_node[0].get("text") != "Title":
        return False
    node = root.xpath(
        '//node[@resource-id="com.google.android.keep:id/edit_note_text"]'
    )
    if not node:
        return False
    if content in node[0].get("text"):
        return True
    return False


@evaluator(env_name="android")
def check_bluetooth_name(content: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    bluetooth_node = root.xpath('//node[@resource-id="android:id/summary"]')
    if not bluetooth_node:
        return False
    if content in bluetooth_node[0].get("text"):
        return True
    return False


@evaluator(env_name="android")
def check_map_direction_page(from_des: str, to_des: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    from_node = root.xpath(f'//node[@content-desc="Start location, {from_des}"]')
    if not from_node:
        return False
    to_node = root.xpath(f'//node[@content-desc="Destination, {to_des}"]')
    if not to_node:
        return False
    return True


@evaluator(env_name="android")
def check_dial_number(phone_number: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    dialer_node = root.xpath('//node[@resource-id="com.android.dialer:id/digits"]')
    if not dialer_node:
        return False
    number = dialer_node[0].get("text")
    number = re.sub("[^0-9]", "", number)
    target = re.sub("[^0-9]", "", phone_number)
    return number == target


@evaluator(env_name="android")
def check_calendar_registered(date: str, content: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    calendar_node = root.xpath(
        '//node[@resource-id="com.google.android.calendar:id/alternate_timeline_fragment_container"]'
    )
    if not calendar_node:
        return False
    itr_calendar_node = calendar_node[0].xpath(
        '//node[@class="android.support.v7.widget.RecyclerView"]'
    )
    if not itr_calendar_node:
        return False
    target_nodes = itr_calendar_node[0].xpath('//node[@content-desc="{content}"]')
    if not target_nodes:
        return False
    return True


@evaluator(env_name="android")
def check_drive_registered(content: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    entry_node = root.xpath(
        '//node[@resource-id="com.google.android.apps.docs:id/entry_label"]'
    )
    if not entry_node:
        return False
    for node in entry_node:
        if content == node.get("text") and f"{content} Folder" == node.get(
            "content-desc"
        ):
            return True
    return False


@evaluator(env_name="android")
def check_contact_registered(mail: str, name: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    name_node = root.xpath('//node[@resource-id="com.android.contacts:id/large_title"]')
    if not name_node:
        return False
    text = name_node[0].get("text")
    if text not in name:
        return False

    mail_node = root.xpath('//node[@resource-id="com.android.contacts:id/header"]')
    text = mail_node[0].get("text")
    if text not in mail:
        return False
    return True


@evaluator(env_name="android")
def check_calling_number(phone_number: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    dialer_node = root.xpath(
        '//node[@resource-id="com.android.dialer:id/contactgrid_contact_name"]'
    )
    if not dialer_node:
        return False
    number = dialer_node[0].get("text")
    number = re.sub("[^0-9]", "", number)
    target = re.sub("[^0-9]", "", phone_number)
    return number == target


@evaluator(env_name="android")
def check_google_tasks_name(target: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    task_nodes = root.xpath(
        '//node[@resource-id="com.google.android.apps.tasks:id/task_name"]'
    )
    if not task_nodes:
        return False
    for node in task_nodes:
        task_name = node.get("text")
        if target in task_name:
            return True
    return False


@evaluator(env_name="android")
def check_date(target: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    date_nodes = root.xpath(
        '//node[@resource-id="com.google.android.apps.photos:id/datetime_item_layout"]'
    )
    if not date_nodes:
        return False
    prev_node = date_nodes.xpath(
        '//node[@resource-id="com.google.android.apps.photos:id/label"]'
    )
    time = prev_node.get("text")
    pattern = re.compile(r"^\w{3},\s\w{3}\s\d{2},\s\d{4}\s•\s\d{1,2}:\d{2}\s[AP]M$")
    if pattern.match(time):
        return True
    return False


@evaluator(env_name="android")
def check_city_clock(place_name: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    city_nodes = root.xpath(
        '//node[@resource-id="com.google.android.deskclock:id/city_name"]'
    )
    if city_nodes is None:
        return False
    for city_node in city_nodes:
        text = city_node.get("text")
        if place_name == text:
            return True
    return False


@evaluator(env_name="android")
def check_event(date: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    event_nodes = root.xpath('//node[@class="android.support.v7.widget.RecyclerView"]')
    if event_nodes is None:
        return False
    if not event_nodes:
        return False
    for node in event_nodes[0]:
        text = node.get("content-desc")
        if date in text:
            return True
    return False


@evaluator(env_name="android")
def check_event_registered(date: str, content: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    event_nodes = root.xpath('//node[@class="android.support.v7.widget.RecyclerView"]')
    if not event_nodes:
        return False
    time_reg = False
    content_reg = False
    for node in event_nodes[0]:
        text = node.get("content-desc")
        if date.lower() in text.lower():
            time_reg = True
        if content.lower() in text.lower():
            content_reg = True
    if time_reg and content_reg:
        return True
    return False


@evaluator(env_name="android")
def check_location(content: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    checked_node = root.xpath(f'//node[@content-desc="{content}"]')
    if not checked_node:
        return False
    return True


@evaluator(env_name="android")
def check_contain_city(number: str, city: str, env) -> bool:
    root = get_xml_etree(env)
    print(root)
    if root is None:
        return False
    business_node = root.xpath(
        '//node[@resource-id="com.google.android.apps.maps:id/search_omnibox_text_box"]'
    )
    if not business_node:
        return False
    text = None
    for node in business_node[0]:
        text = node.get("text")
    if text is None:
        return False
    if city in text and str(number) in text:
        return True
    return False


@evaluator(env_name="android")
def check_file(content: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    name_source_node = root.xpath(
        '//node[@resource-id="com.google.android.apps.photos:id/exif_item_layout"]'
    )
    if not name_source_node:
        return False
    name_nodes = name_source_node[0].xpath(
        '//node[@resource-id="com.google.android.apps.photos:id/label"]'
    )
    if not name_nodes:
        return False
    target_node = None
    for node in name_nodes:
        text = node.get("text")
        if content in text:
            target_node = node
    if target_node is None:
        return False
    time_source_node = root.xpath(
        '//node[@resource-id="com.google.android.apps.photos:id/datetime_item_layout"]'
    )
    if not time_source_node:
        return False
    time_nodes = time_source_node[0].xpath(
        '//node[@resource-id="com.google.android.apps.photos:id/label"]'
    )
    if not time_nodes:
        return False
    target_node = None
    for node in time_nodes:
        text = node.get("text")
        pattern = re.compile(
            r"(Tue|Mon|Wed|Thu|Fri|Sat|Sun),\s(May|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{2},\s\d{4} • \d{2}:\d{2}\s(AM|PM)"
        )
        if pattern.match(text):
            return True
        return False


@evaluator(env_name="android")
def check_mail_sent(mail: str, content: str, env) -> bool:
    root = get_xml_etree(env)
    if root is None:
        return False
    to_node = root.xpath(
        '//node[@resource-id="com.google.android.gm:id/peoplekit_chip"]'
    )
    if not to_node:
        return False
    checked = False
    for node in to_node:
        text = node.get("content-desc")
        if mail in text:
            checked = True
    if not checked:
        return False
    # check the mail information-> Done

    # check the content information
    body_node = root.xpath(
        '//node[@resource-id="com.google.android.gm:id/body_wrapper"]'
    )
    if not body_node:
        return False
    text_node = body_node[0].xpath('//node[@class="android.widget.EditText"]')
    if not text_node:
        return False
    for node in text_node:
        text = node.get("text")
        if content in text:
            return True
    return False
##########################################################
###################下面是新造的evaluator
@evaluator(env_name="android", local=True)
def check_task_reminder(env) -> bool:
    print("check_task_reminder")
    root=get_xml_etree(env)
    if root is None:
        return False
    # 假设 root 是已经加载的 XML 文档
    nodes = root.xpath('//node[@class="android.widget.TextView" and contains(@text, "任务提醒")]')
    if not nodes:
        return False
    return True
#检查是否进入小雅某个模块界面
@evaluator(env_name="android", local=True)
def check_xiaoya_module(module_name,env) -> bool:
    print("check_xiaoya_module", module_name)
    root=get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath(f'//node[@class="android.widget.TextView" and contains(@text, "{module_name}")]')
    for node in nodes:
        # 检查父节点的 class 属性
        parent = node.getparent()
        if parent.get("index") == "1":
            return True
    return False
##检查是否进入小雅某个课程界面
@evaluator(env_name="android", local=True)
def check_xiaoya_course(course_name,env) -> bool:
    print("check_xiaoya_course", course_name)
    root=get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath(f'//node[@class="android.widget.TextView" and contains(@text, "{course_name}")]')
    for node in nodes:
        # 检查父节点的 class 属性
        parent = node.getparent()
        if parent.get("index") == "5":
            return True
    return False

#检查是否进入QQ联系人界面
@evaluator(env_name="android", local=True)
def check_qq_contact(env) -> bool:
    print("check_qq_contact")
    root = get_xml_etree(env)
    if root is None:
        return False

    # 查找符合条件的节点
    nodes = root.xpath('//node[@resource-id="com.tencent.mobileqq:id/ivTitleName"]')

    # 检查节点是否存在且文本为"联系人"
    if nodes and nodes[0].get("text") == "联系人":
        return True
    return False

@evaluator(env_name="android", local=True)
def check_huashi_course(env) -> bool:
    print("check_huashi_course")
    root = get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath('//node[@resource-id="net.muxi.huashiapp:id/tv_select_week"]')
    if nodes:
        return True
    return False


@evaluator(env_name="android", local=True)
def check_huashi_grade(env) -> bool:
    print("check_huashi_grade")
    root = get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath(f'//node[@class="android.widget.TextView" and contains(@text, "查算学分绩")]')
    for node in nodes:
        # 检查父节点的 class 属性
        parent = node.getparent()
        if parent.get("index") == "0":
            return True
    return False

@evaluator(env_name="android", local=True)
def check_huashi_course_grade(env) -> bool:
    print("check_huashi_course_grade")
    root = get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath(f'//node[@resource-id="net.muxi.huashiapp:id/tv_course_name"]')
    if nodes:
        return True
    return False

@evaluator(env_name="android", local=True)
def check_calculate_grade(env) -> bool:
    print("check_calculate_grade")
    root = get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath(f'//node[@class="android.widget.TextView" and contains(@text, "计算结果")]')
    if nodes:
        return True
    return False


@evaluator(env_name="android", local=True)
def check_huashi_xiaoli(env) -> bool:
    print("check_huashi_xiaoli")
    root = get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath(f'//node[@class="android.widget.TextView" and contains(@text, "校历")]')
    for node in nodes:
        # 检查父节点的 class 属性
        parent = node.getparent()
        if parent.get("index") == "0":
            return True
    return False

@evaluator(env_name="android", local=True)
def check_huashi_module(modeule_name,env) -> bool:
    print("check_huashi_module", modeule_name)
    root = get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath(f'//node[@class="android.widget.TextView" and contains(@text, "{modeule_name}")]')
    for node in nodes:
        # 检查父节点的 class 属性
        parent = node.getparent()
        if parent.get("index") == "0":
            return True
    return False

#检查是否进入华师匣子的更多页面
@evaluator(env_name="android", local=True)
def check_huashi_more(env)->bool:
    print("check_huashi_more")
    root = get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath(f'//node[@class="android.widget.TextView" and contains(@text,"更多")]')
    for node in nodes:
        parent = node.getparent()
        if parent.get("index") == "0":
            return True
    return False

#检查是否进入意见反馈页面
@evaluator(env_name="android", local=True)
def check_huashi_more_module(module_name,env)->bool:
    print("check_huashi_more_module", module_name)
    root = get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath(f'//node[@class="android.widget.TextView" and contains(@text, "{module_name}")]')
    for node in nodes:
        parent = node.getparent()
        if parent.get("index") == "0":
            return True
    return False
#检查是否完成反馈
@evaluator(env_name="android", local=True)
def check_huashi_advice_complete(env)->bool:
    print("check_huashi_advice_complete")
    root = get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath(f'//node[@resource-id="net.muxi.huashiapp:id/title_sugg"]')
    if nodes:
        return True
    return False

@evaluator(env_name="android", local=True)
def check_huashi_site(site_name,env)->bool:
    print("check_huashi_site", site_name)
    root = get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath(f'//node[@class="android.widget.TextView" and contains(@text, "{site_name}")]')
    for node in nodes:
        if node.get("index")=="1":
            return True
    return False

@evaluator(env_name="android", local=True)
def check_huashi_classroom(env)->bool:
    print("check_huashi_classroom")
    root = get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath(f'//node[@class="android.widget.TextView" and contains(@text, "空闲教室列表")]')
    for node in nodes:
        if node.get("index") == "1":
            return True
    return False

@evaluator(env_name="android", local=True)
def check_huashi_close_notice(content,env)->bool:
    print("check_huashi_close_notice",content)
    root = get_xml_etree(env)
    if root is None:
        return False
    if content=="通知栏提醒":
        nodes = root.xpath(f'//node[@resource-id="net.muxi.huashiapp:id/switch_course_remind"]')
    elif content=="图书提醒":
        nodes = root.xpath(f'//node[@resource-id="net.muxi.huashiapp:id/switch_library_remind"]')
    elif content=="学生卡消息提醒":
        nodes = root.xpath(f'//node[@resource-id="net.muxi.huashiapp:id/switch_card_remind"]')
    elif content=="成绩消息提醒":
        nodes = root.xpath(f'//node[@resource-id="net.muxi.huashiapp:id/switch_score_remind"]')

    for node in nodes:
        if node.get("text")=="关闭":
            return  True
    return  False

#判断是否进入了消息模块
@evaluator(env_name="android", local=True)
def check_xiaoya_message(name,env) -> bool:
    print("check_xiaoya_message",name)
    root=get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath(f'//node[@class="android.widget.TextView" and contains(@text, "{name}")]')
    for node in nodes:
        # 检查父节点的 class 属性
        parent = node.getparent()
        if parent.get("index") == "0":
            return True
    return False

#检查是否进入了作业任务、课程工具、分组等
@evaluator(env_name="android", local=True)
def check_tasktoolandsoon(name,env) -> bool:
    print("check tasktoolandsoon")
    root=get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath(f'//node[@class="android.widget.TextView" and contains(@text, "{name}")]')
    for node in nodes:
        # 检查父节点的 class 属性
        parent = node.getparent()
        if parent.get("index") == "1":
            return True
    return False
#判断是否进入了错题本
@evaluator(env_name="android", local=True)
def check_mistake(env) -> bool:
    print("check mistake")
    root=get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath(f'//node[@class="android.widget.TextView" and contains(@text, "错题本")]')
    for node in nodes:
        # 检查父节点的 class 属性
        parent = node.getparent()
        if parent.get("index") == "9":
            return True
    return False
#学生评教
@evaluator(env_name="android", local=True)
def student_evaluate(env) -> bool:
    print("check studentevaluate")
    root=get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath(f'//node[@class="android.widget.TextView" and contains(@text, "学生评教")]')
    for node in nodes:
        # 检查父节点的 class 属性
        parent = node.getparent()
        if parent.get("index") == "6":
            return True
    return False

@evaluator(env_name="android", local=True)
def student_assistant(env) -> bool:
    print("check assistant")
    root=get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath(f'//node[@class="android.widget.TextView" and contains(@text, "AI助手")]')
    for node in nodes:
        # 检查父节点的 class 属性
        parent = node.getparent()
        if parent.get("index") == "1":
            return True
    return False

@evaluator(env_name="android", local=True)
def has_asked_question(env) -> bool:
    print("check question")
    root=get_xml_etree(env)
    if root is None:
        return False
    edit_text = root.find('.//node[@class="android.widget.EditText"]')
    if edit_text is not None and edit_text.get("text", "").strip():
        return True
# 检查发送按钮状态
    send_button = root.find('.//node[@text="发送" and @class="android.widget.Button"]')
    if send_button is not None and send_button.get("enabled", "false") == "true":
        return True

    # 检查消息记录
    message_nodes = root.findall('.//node[@class="android.widget.TextView"]')
    for node in message_nodes:
        if node.get("text", "").strip():
            return True

    return False

@evaluator(env_name="android", local=True)
def check_xiaoya_wode(env) -> bool:
    print("check_xiaoya_wode")
    root=get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath(f'//node[@class="android.widget.TextView" and contains(@text, "退出小雅")]')
    for node in nodes:
        # 检查父节点的 class 属性
        parent = node.getparent()
        if parent.get("index") == "0":
            return True
    return False

@evaluator(env_name="android", local=True)
def check_xiaoya_selfreport(env) -> bool:
    print("check_xiaoya_selfreport")
    root=get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath(f'//node[@class="android.webkit.WebView" and contains(@text, "个人年度报告-2024")]')
    for node in nodes:
        # 检查父节点的 class 属性
        parent = node.getparent()
        if parent.get("index") == "0":
            return True
    return False


@evaluator(env_name="android", local=True)
def check_entergroup(env) -> bool:
    print("check group")
    root=get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath(f'//node[@class="android.widget.TextView" and contains(@text, "加入小组")]')
    for node in nodes:
        # 检查父节点的 class 属性
        parent = node.getparent()
        if parent.get("index") == "1":
            return True
    return False

@evaluator(env_name="android", local=True)
def find_teacher_contact(name,env) -> bool:
    print("find teacher contact number",name)
    root=get_xml_etree(env)
    if root is None:
        return False
    nodes = root.xpath(f'//node[@class="android.widget.TextView" and contains(@text, "{name}")]')
    for node in nodes:
        # 检查父节点的 class 属性
        parent = node.getparent()
        if parent.get("index") == "1":
            return True
    return False

#################到此为止###################################


def mail_evaluator_generator(mail: str, content: str):
    result = nx.DiGraph()
    a = check_current_package_name("com.google.android.gm")
    b = check_contain_input_text(mail)
    c = check_contain_input_text(content)
    d = check_mail_sent(mail, content)
    result.add_edges_from([(a, b), (a, c), (b, d), (c, d)])
    return result


android_subtasks = [
    SubTask(
        id="a3d11574-2acf-4b26-a569-a5dbc9d548ah",
        description='In Android, Using "Clock" app, set the time of {place_name} in the clock, check the time gap between the city and current city.',
        attribute_dict={"place_name": "place_name"},
        output_type="content",
        evaluator_generator=lambda place_name: path_graph(
            [
                check_current_package_name("com.google.android.deskclock"),
                check_city_clock(place_name),
            ],
            create_using=DiGraph,
        ),
    ),
    ########################################################
    ###################下面均为新的任务#######################
    SubTask(
        id="2394b768-2ca7-45e9-b41e-2aa4e9573192",
        description='In android system, use the calendar app, find the title of an event in the date "{date}".',
        attribute_dict={"date": "date"},
        output_type="content",
        evaluator_generator=lambda date: path_graph(
            [
                check_current_package_name("com.google.android.calendar"),
                check_event(date),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="a3d11574-2acf-4b26-a569-a5dbc9d548af",
        description='Using "Tasks" app, add a new task with text "{content}".',
        attribute_dict={"content": "content"},
        output_type="None",
        evaluator_generator=lambda content: path_graph(
            [
                check_current_package_name("com.google.android.apps.tasks"),
                check_contain_input_text(content),
                check_google_tasks_name(content),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="eb92a1e6-4c86-4d56-baac-95fc8397732e",
        description='In Android, using "Keep Notes" App, record "{content}" in a new note without title.',
        attribute_dict={"content": "content"},
        output_type="None",
        evaluator_generator=lambda content: path_graph(
            [
                check_current_package_name("com.google.android.keep"),
                check_contain_input_text(content),
                check_note_content(content),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="a3d11574-2acf-4b26-a569-a5dbc9d548ac",
        description='In Android, Using "Calendar" app, add a new event with text "{content}" in date "{date}" all day.',
        attribute_dict={"content": "content", "date": "date"},
        output_type="None",
        evaluator_generator=lambda content, date: path_graph(
            [
                check_current_package_name("com.google.android.calendar"),
                check_contain_input_text(content),
                check_event_registered(date, content),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="0090f116-e02b-4562-a20d-b5df38be963a",
        description='In Android, Using "Gmail" app, send {mail} a message {content}.',
        attribute_dict={"content": "content", "mail": "mail"},
        output_type="None",
        evaluator_generator=mail_evaluator_generator,
    ),
    SubTask(
        id="open_xiaoya_web",
        description="In Android, open the web 'https://www.ai-augmented.com/'",
        attribute_dict={},
        output_type="None",
        evaluator_generator=lambda package_name: path_graph(
            [
                check_current_package_name("com.android.chrome"),
                check_contain_input_text('https://www.ai-augmented.com/'),
                check_contain_input_text('智能助手')
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="huashi_app_course",
        description="查看华师匣子中的课程表",
        attribute_dict={},
        output_type="None",
        evaluator_generator=lambda : path_graph(
            [
                check_current_package_name("net.muxi.huashiapp"),
                check_huashi_course()
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="huashi_app_grade",
        description="查看华师匣子中的成绩并计算平均学分绩",
        attribute_dict={},
        output_type="content",
        evaluator_generator=lambda : path_graph(
            [
                check_current_package_name("net.muxi.huashiapp"),
                check_huashi_grade(),
                check_huashi_course_grade(),
                check_calculate_grade()
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="huashi_app_xiaoli",
        description="查看华师匣子中的校历",
        attribute_dict={},
        output_type="None",
        evaluator_generator=lambda : path_graph(
            [
                check_current_package_name("net.muxi.huashiapp"),
                check_huashi_xiaoli(),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="huashi_app_advice",
        description="在华师匣子中写意见反馈",
        attribute_dict={"content":"content"},
        output_type="None",
        evaluator_generator=lambda content: path_graph(
            [
                check_current_package_name("net.muxi.huashiapp"),
                check_huashi_more(),
                check_contain_input_text(content),
                check_huashi_advice_complete(),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="huashi_app_site",
        description="在华师匣子的常用网站访问网站{site}",
        attribute_dict={"site":"site"},
        output_type="None",
        evaluator_generator=lambda site: path_graph(
            [
                check_current_package_name("net.muxi.huashiapp"),
                check_huashi_module("常用网站"),
                check_huashi_site(site)
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="huashi_app_classroom",
        description="在华师匣子中查询空闲教室",
        attribute_dict={},
        output_type="None",
        evaluator_generator=lambda : path_graph(
            [
                check_current_package_name("net.muxi.huashiapp"),
                check_huashi_module("空闲教室"),
                check_huashi_classroom()
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
        id="huashi_app_add_course",
        description="在华师匣子中添加新课程",
        attribute_dict={"course":"course"},
        output_type="None",
        evaluator_generator=lambda course : path_graph(
            [
                check_current_package_name("net.muxi.huashiapp"),
                check_huashi_course(),
                check_contain_input_text(course)
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(
            id="check_some_course",
            description='查看我的课程里的课程"{course}"',
            attribute_dict={"course":"course"},
            output_type="None",
            evaluator_generator=lambda course: path_graph(
                [
                    check_current_package_name("com.ccnu.jx.xiaoya"),
                    check_xiaoya_module("课程"),
                    check_xiaoya_course(course)
                ],
                create_using=DiGraph,
            ),
        ),
    SubTask(
            id="xiaoya_write_document",
            description='在小雅里我的文档中新建笔记，并写入内容{content}',
            attribute_dict={"content":"content"},
            output_type="None",
            evaluator_generator=lambda content: path_graph(
                [
                    check_current_package_name("com.ccnu.jx.xiaoya"),
                    check_xiaoya_module("我的文档"),
                    check_contain_input_text(content)
                ],
                create_using=DiGraph,
            ),
        ),
    SubTask(
            id="check_undone_works",
            description='在安卓，使用‘小雅智能助’应用查看待完成的任务数量',
            attribute_dict={},
            output_type="content",
            evaluator_generator=lambda : path_graph(
                [
                    check_current_package_name("com.ccnu.jx.xiaoya"),
                    check_task_reminder(),
                    # check_contain_text("待完成学习任务"),
                ],
                create_using=DiGraph,
            ),
        ),

    SubTask(  # test:ok
        id="check_some_course",
        description='在小雅中查看"{course}"的课程资料和教学内容',
        attribute_dict={"course": "course"},
        output_type="None",
        evaluator_generator=lambda course: path_graph(
            [
                check_current_package_name("com.ccnu.jx.xiaoya"),
                check_xiaoya_module("课程"),
                check_xiaoya_course(course)
            ],
            create_using=DiGraph,
        ),
    ),

    SubTask(  # test:  ok
        id="check_course_task",
        description='在小雅智能助中查看"{course}"的作业任务',
        attribute_dict={"course": "course"},
        output_type="None",
        evaluator_generator=lambda course: path_graph(
            [
                check_current_package_name("com.ccnu.jx.xiaoya"),
                check_xiaoya_module("课程"),
                check_xiaoya_course(course),
                check_tasktoolandsoon("任务")
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(  # test: ok
        id="check_course_mistake",
        description='在小雅智能助中查看"{course}"的错题本',
        attribute_dict={"course": "course"},
        output_type="None",
        evaluator_generator=lambda course: path_graph(
            [
                check_current_package_name("com.ccnu.jx.xiaoya"),
                check_xiaoya_module("课程"),
                check_xiaoya_course(course),
                check_tasktoolandsoon("课程工具"),
                check_mistake()
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(  # test: ok
        id="course_tool-students evaluate",
        description='在小雅智能助中进入"{course}"的学生评教',
        attribute_dict={"course": "course"},
        output_type="None",
        evaluator_generator=lambda course: path_graph(
            [
                check_current_package_name("com.ccnu.jx.xiaoya"),
                check_xiaoya_module("课程"),
                check_xiaoya_course(course),
                check_tasktoolandsoon("课程工具"),
                student_evaluate()
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(  # test: ok
        id="course_tool-students assistant",
        description='在小雅智能助中查看"{course}"的课程工具中的智能助手',
        attribute_dict={"course": "course"},
        output_type="None",
        evaluator_generator=lambda course: path_graph(
            [
                check_current_package_name("com.ccnu.jx.xiaoya"),
                check_xiaoya_module("课程"),
                check_xiaoya_course(course),
                check_tasktoolandsoon("课程工具"),
                student_assistant(),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(  # test: ok
        id="add-group",
        description='在小雅智能助中"{course}"的分组并加入分组',
        attribute_dict={"course": "course"},
        output_type="None",
        evaluator_generator=lambda course: path_graph(
            [
                check_current_package_name("com.ccnu.jx.xiaoya"),
                check_xiaoya_module("课程"),
                check_xiaoya_course(course),
                check_tasktoolandsoon("小组"),
            ],
            create_using=DiGraph,
        ),
    ),
    SubTask(  # test: ok
        id="find_contact",
        description='在小雅消息中心中的通讯录查看"{course}"的老师联系方式',
        attribute_dict={"course": "course"},
        output_type="number",
        evaluator_generator=lambda course: path_graph(
            [
                check_current_package_name("com.ccnu.jx.xiaoya"),
                check_xiaoya_message("消息中心"),
                check_xiaoya_module("通讯录"),
                find_teacher_contact(course)
            ],
            create_using=DiGraph,
        ),

    ),
SubTask( id="51b2463c-9904-4a32-81ba-507bfb89d61f", description='In Android, Using "Google Map" app, Find the city name of corresponding post code "{number}" in the country "{country}".', attribute_dict={"number": "number", "country": "country"}, output_type="content", evaluator_generator=lambda number, country: path_graph( [ check_current_package_name("com.google.android.apps.maps"), check_contain_input_text(country), check_contain_input_text(number), check_contain_city(number, country), ], create_using=DiGraph, ), ),
    SubTask(  # test: ok
        id="find_course_message",
        description='在小雅消息中心中的课程消息中查看课程相关消息',
        attribute_dict={},
        output_type="number",
        evaluator_generator=lambda: path_graph(
            [
                check_current_package_name("com.ccnu.jx.xiaoya"),
                check_xiaoya_message("消息中心"),
                check_xiaoya_module("课程消息"),
            ],
            create_using=DiGraph,
        ),
    ),

    # TODO: The phone number page cannot be accesed by xml. figure out another way.
    # SubTask(
    #     id="fa9c0b01-9835-4932-824d-0990cb20e5f7",
    #     description='Using Settings app, find the phone number of this phone in the "About" panel.',
    #     attribute_dict={},
    #     output_type="phone_number",
    #     evaluator=lambda: path_graph(
    #         [
    #             check_current_package_name("com.android.settings"),
    #         ],
    #         create_using=DiGraph,
    #     ),
    # ),
]
